import math
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def gdn_prefill_kernel(
    # Input pointers
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    cu_seqlens_ptr, scale,
    # Output pointers
    output_ptr, new_state_ptr,
    # Strides for q [total_seq_len, num_q_heads, head_size]
    stride_q_t, stride_q_h, stride_q_d,
    # Strides for k [total_seq_len, num_k_heads, head_size]
    stride_k_t, stride_k_h, stride_k_d,
    # Strides for v [total_seq_len, num_v_heads, head_size]
    stride_v_t, stride_v_h, stride_v_d,
    # Strides for state [num_seqs, num_v_heads, head_size, head_size] (V-last in storage but k-last means [N,H,V,K])
    stride_s_n, stride_s_h, stride_s_v, stride_s_k,
    # Strides for a [total_seq_len, num_v_heads]
    stride_a_t, stride_a_h,
    # Strides for b [total_seq_len, num_v_heads]
    stride_b_t, stride_b_h,
    # Strides for output [total_seq_len, num_v_heads, head_size]
    stride_o_t, stride_o_h, stride_o_d,
    # Dimensions
    num_q_heads, num_k_heads, num_v_heads,
    H_K: tl.constexpr,  # head_size = 128
    BV: tl.constexpr,   # block size over V dimension
    has_state: tl.constexpr,
):
    """
    GDN prefill kernel.
    Grid: (num_seqs, num_v_heads, cdiv(H_K, BV))
    Each block handles one sequence, one v_head, one tile of V dimension.
    Processes tokens sequentially, updating state tile.
    """
    i_seq = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_bv = tl.program_id(2)

    # GVA mapping: v_head -> q/k head
    gva_ratio = num_v_heads // num_q_heads  # = 2
    i_hq = i_hv // gva_ratio

    # Get sequence bounds
    seq_start = tl.load(cu_seqlens_ptr + i_seq)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1)
    seq_len = seq_end - seq_start

    if seq_len <= 0:
        return

    # V tile offsets (this block handles rows i_bv*BV .. i_bv*BV+BV-1 of V)
    offs_v = i_bv * BV + tl.arange(0, BV)
    v_mask = offs_v < H_K  # V dimension = head_size = H_K

    # K offsets (full K dimension)
    offs_k = tl.arange(0, H_K)

    # Load A_log for this v_head
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    A_val = tl.exp(A_log_val)  # exp(A_log)

    # Load dt_bias for this v_head
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    # Load initial state tile [BV, H_K] from state[i_seq, i_hv, :, :]
    # State layout: [N, H_v, V, K] -> state[i_seq, i_hv, offs_v, offs_k]
    state_base = i_seq * stride_s_n + i_hv * stride_s_h
    s_ptrs = state_ptr + state_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    s_mask = v_mask[:, None]  # [BV, 1] broadcast over K

    if has_state:
        state_tile = tl.load(s_ptrs, mask=s_mask, other=0.0)  # [BV, H_K] float32
    else:
        state_tile = tl.zeros([BV, H_K], dtype=tl.float32)

    # Process tokens sequentially
    for i in range(seq_len):
        t = seq_start + i

        # Load k vector for this token [H_K] (using q/k head)
        k_base = t * stride_k_t + i_hq * stride_k_h
        k_vec = tl.load(k_ptr + k_base + offs_k * stride_k_d).to(tl.float32)  # [H_K]

        # Load v vector for this token [BV] (using v head, tile)
        v_base = t * stride_v_t + i_hv * stride_v_h
        v_vec = tl.load(v_ptr + v_base + offs_v * stride_v_d, mask=v_mask, other=0.0).to(tl.float32)  # [BV]

        # Load a for this token/v_head, compute g
        a_val = tl.load(a_ptr + t * stride_a_t + i_hv * stride_a_h).to(tl.float32)
        x_val = a_val + dt_bias_val
        # softplus(x) = log(1 + exp(x))
        softplus_x = tl.log(1.0 + tl.exp(x_val))
        g_val = tl.exp(-A_val * softplus_x)  # scalar

        # Load b for this token/v_head, compute beta
        b_val = tl.load(b_ptr + t * stride_b_t + i_hv * stride_b_h).to(tl.float32)
        beta_val = tl.sigmoid(b_val)  # scalar

        # Decay state
        old_state = g_val * state_tile  # [BV, H_K]

        # Compute old_v = state @ k = sum over K dim: old_state[bv, k] * k_vec[k]
        old_v = tl.sum(old_state * k_vec[None, :], axis=1)  # [BV]

        # new_v = beta * v + (1 - beta) * old_v
        new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [BV]

        # State update: state = old_state - k^T @ old_v + k^T @ new_v
        # = old_state + k^T @ (new_v - old_v)
        # delta_v = new_v - old_v = beta * (v - old_v)
        delta_v = new_v - old_v  # [BV]

        # state += delta_v[:, None] * k_vec[None, :]
        state_tile = old_state + delta_v[:, None] * k_vec[None, :]  # [BV, H_K]

        # Compute output: o = scale * q @ state = scale * sum_k(q[k] * state[:, k])
        # Load q vector
        q_base = t * stride_q_t + i_hq * stride_q_h
        q_vec = tl.load(q_ptr + q_base + offs_k * stride_q_d).to(tl.float32)  # [H_K]

        # o[bv] = scale * sum_k(q_vec[k] * state_tile[bv, k])
        o_vec = scale * tl.sum(state_tile * q_vec[None, :], axis=1)  # [BV]

        # Store output
        out_base = t * stride_o_t + i_hv * stride_o_h
        tl.store(output_ptr + out_base + offs_v * stride_o_d,
                 o_vec.to(tl.bfloat16), mask=v_mask)

    # Store final state tile
    new_state_base = i_seq * stride_s_n + i_hv * stride_s_h
    ns_ptrs = new_state_ptr + new_state_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    tl.store(ns_ptrs, state_tile, mask=s_mask)


@triton.jit
def gdn_prefill_kernel_chunk(
    # Input pointers
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    cu_seqlens_ptr, scale,
    # Output pointers
    output_ptr, new_state_ptr,
    # Strides for q [total_seq_len, num_q_heads, head_size]
    stride_q_t, stride_q_h, stride_q_d,
    # Strides for k [total_seq_len, num_k_heads, head_size]
    stride_k_t, stride_k_h, stride_k_d,
    # Strides for v [total_seq_len, num_v_heads, head_size]
    stride_v_t, stride_v_h, stride_v_d,
    # Strides for state [num_seqs, num_v_heads, head_size, head_size]
    stride_s_n, stride_s_h, stride_s_v, stride_s_k,
    # Strides for a [total_seq_len, num_v_heads]
    stride_a_t, stride_a_h,
    # Strides for b [total_seq_len, num_v_heads]
    stride_b_t, stride_b_h,
    # Strides for output [total_seq_len, num_v_heads, head_size]
    stride_o_t, stride_o_h, stride_o_d,
    # Dimensions
    num_q_heads, num_k_heads, num_v_heads,
    H_K: tl.constexpr,
    BV: tl.constexpr,
    CHUNK: tl.constexpr,
    has_state: tl.constexpr,
):
    """
    GDN prefill kernel with chunked processing.
    Grid: (num_seqs, num_v_heads, cdiv(H_K, BV))
    """
    i_seq = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_bv = tl.program_id(2)

    gva_ratio = num_v_heads // num_q_heads
    i_hq = i_hv // gva_ratio

    seq_start = tl.load(cu_seqlens_ptr + i_seq)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1)
    seq_len = seq_end - seq_start

    if seq_len <= 0:
        return

    offs_v = i_bv * BV + tl.arange(0, BV)
    v_mask = offs_v < H_K
    offs_k = tl.arange(0, H_K)

    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    A_val = tl.exp(A_log_val)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    state_base = i_seq * stride_s_n + i_hv * stride_s_h
    s_ptrs = state_ptr + state_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    s_mask = v_mask[:, None]

    if has_state:
        state_tile = tl.load(s_ptrs, mask=s_mask, other=0.0)
    else:
        state_tile = tl.zeros([BV, H_K], dtype=tl.float32)

    num_chunks = tl.cdiv(seq_len, CHUNK)

    for chunk_idx in range(num_chunks):
        chunk_start = seq_start + chunk_idx * CHUNK
        chunk_end = tl.minimum(chunk_start + CHUNK, seq_end)
        chunk_len = chunk_end - chunk_start

        for i in range(CHUNK):
            if i < chunk_len:
                t = chunk_start + i

                k_base = t * stride_k_t + i_hq * stride_k_h
                k_vec = tl.load(k_ptr + k_base + offs_k * stride_k_d).to(tl.float32)

                v_base = t * stride_v_t + i_hv * stride_v_h
                v_vec = tl.load(v_ptr + v_base + offs_v * stride_v_d, mask=v_mask, other=0.0).to(tl.float32)

                a_val = tl.load(a_ptr + t * stride_a_t + i_hv * stride_a_h).to(tl.float32)
                x_val = a_val + dt_bias_val
                softplus_x = tl.log(1.0 + tl.exp(x_val))
                g_val = tl.exp(-A_val * softplus_x)

                b_val = tl.load(b_ptr + t * stride_b_t + i_hv * stride_b_h).to(tl.float32)
                beta_val = tl.sigmoid(b_val)

                old_state = g_val * state_tile
                old_v = tl.sum(old_state * k_vec[None, :], axis=1)
                delta_v = beta_val * (v_vec - old_v)
                state_tile = old_state + delta_v[:, None] * k_vec[None, :]

                q_base = t * stride_q_t + i_hq * stride_q_h
                q_vec = tl.load(q_ptr + q_base + offs_k * stride_q_d).to(tl.float32)
                o_vec = scale * tl.sum(state_tile * q_vec[None, :], axis=1)

                out_base = t * stride_o_t + i_hv * stride_o_h
                tl.store(output_ptr + out_base + offs_v * stride_o_d,
                         o_vec.to(tl.bfloat16), mask=v_mask)

    new_state_base = i_seq * stride_s_n + i_hv * stride_s_h
    ns_ptrs = new_state_ptr + new_state_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    tl.store(ns_ptrs, state_tile, mask=s_mask)


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill with GVA configuration and k-last state layout.
    
    Args:
        q: [total_seq_len, num_q_heads, head_size] bfloat16
        k: [total_seq_len, num_k_heads, head_size] bfloat16
        v: [total_seq_len, num_v_heads, head_size] bfloat16
        state: [num_seqs, num_v_heads, head_size, head_size] float32 or None (k-last: [N,H,V,K])
        A_log: [num_v_heads] float32
        a: [total_seq_len, num_v_heads] bfloat16
        dt_bias: [num_v_heads] float32
        b: [total_seq_len, num_v_heads] bfloat16
        cu_seqlens: [num_seqs+1] int64
        scale: float32 scalar
    
    Returns:
        output: [total_seq_len, num_v_heads, head_size] bfloat16
        new_state: [num_seqs, num_v_heads, head_size, head_size] float32
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)
    scale = float(scale)

    # Allocate outputs
    output = torch.zeros(total_seq_len, num_v_heads, head_size, dtype=torch.bfloat16, device=device)
    new_state = torch.zeros(num_seqs, num_v_heads, head_size, head_size, dtype=torch.float32, device=device)

    if num_seqs == 0 or total_seq_len == 0:
        return output, new_state

    has_state = state is not None

    # Choose BV based on head_size
    # head_size = 128, so we tile over V (=128) dimension
    BV = 64  # Process 64 rows of V at a time -> 2 tiles per head

    # Grid: (num_seqs, num_v_heads, cdiv(head_size, BV))
    num_v_tiles = triton.cdiv(head_size, BV)
    grid = (num_seqs, num_v_heads, num_v_tiles)

    # Use the simpler sequential kernel
    gdn_prefill_kernel[grid](
        q, k, v,
        state if has_state else new_state,  # dummy if no state
        A_log, a, dt_bias, b,
        cu_seqlens, scale,
        output, new_state,
        # q strides
        q.stride(0), q.stride(1), q.stride(2),
        # k strides
        k.stride(0), k.stride(1), k.stride(2),
        # v strides
        v.stride(0), v.stride(1), v.stride(2),
        # state strides (new_state if no initial state)
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        # a strides
        a.stride(0), a.stride(1),
        # b strides
        b.stride(0), b.stride(1),
        # output strides
        output.stride(0), output.stride(1), output.stride(2),
        # dims
        num_q_heads, num_k_heads, num_v_heads,
        head_size,  # H_K constexpr
        BV,         # BV constexpr
        has_state,  # has_state constexpr
    )

    return output, new_state