import math
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def gdn_prefill_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, output_ptr, new_state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, cu_seqlens_ptr,
    scale,
    # Strides for q [total_seq_len, H_q, K]
    stride_qt, stride_qh, stride_qk,
    # Strides for k [total_seq_len, H_k, K]
    stride_kt, stride_kh, stride_kk,
    # Strides for v [total_seq_len, H_v, V]
    stride_vt, stride_vh, stride_vv,
    # Strides for state [N, H_v, V, K]
    stride_sn, stride_sh, stride_sv, stride_sk,
    # Strides for output [total_seq_len, H_v, V]
    stride_ot, stride_oh, stride_ov,
    # Strides for a [total_seq_len, H_v]
    stride_at, stride_ah,
    # Strides for b [total_seq_len, H_v]
    stride_bt, stride_bh,
    # Dims
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    has_state: tl.constexpr,
):
    """
    GDN Prefill kernel.
    Grid: (seq_idx, h_v, v_tile)
    Each block handles one sequence, one v_head, one tile of V dimension.
    Processes tokens sequentially (required for recurrence).
    """
    i_seq = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_vt = tl.program_id(2)

    # GVA mapping
    i_hq = i_hv // GVA_RATIO

    # Get sequence bounds
    seq_start = tl.load(cu_seqlens_ptr + i_seq)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1)
    seq_len = seq_end - seq_start

    if seq_len <= 0:
        return

    # V tile offsets
    offs_v = i_vt * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < V
    offs_k = tl.arange(0, K)

    # Load initial state tile [BLOCK_V, K]
    if has_state:
        state_base = i_seq * stride_sn + i_hv * stride_sh
        s_ptrs = state_ptr + state_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
        state_tile = tl.load(s_ptrs, mask=v_mask[:, None], other=0.0)
    else:
        state_tile = tl.zeros([BLOCK_V, K], dtype=tl.float32)

    # Load A_log, dt_bias once (constant per head)
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    # Process tokens sequentially
    for i in range(seq_len):
        t = seq_start + i

        # Load q for this head [K]
        q_base = t * stride_qt + i_hq * stride_qh
        q = tl.load(q_ptr + q_base + offs_k).to(tl.float32)

        # Load k for this head [K]
        k_base = t * stride_kt + i_hq * stride_kh
        k = tl.load(k_ptr + k_base + offs_k).to(tl.float32)

        # Load v for this v_head, this tile [BLOCK_V]
        v_base = t * stride_vt + i_hv * stride_vh
        v_vals = tl.load(v_ptr + v_base + offs_v * stride_vv, mask=v_mask, other=0.0).to(tl.float32)

        # Load a, b for this token and v_head
        a_val = tl.load(a_ptr + t * stride_at + i_hv * stride_ah).to(tl.float32)
        b_val = tl.load(b_ptr + t * stride_bt + i_hv * stride_bh).to(tl.float32)

        # Compute g = exp(-exp(A_log) * softplus(a + dt_bias))
        x = a_val + dt_bias_val
        # Numerically stable softplus: softplus(x) = log(1 + exp(x))
        # Use stable form to avoid overflow for large x
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-tl.exp(A_log_val) * softplus_x)

        # Compute beta = sigmoid(b)
        beta = tl.sigmoid(b_val)

        # Apply decay to state
        state_tile = g * state_tile

        # Compute state_tile @ k -> [BLOCK_V] (this is k @ state in K-V space)
        k_dot_s = tl.sum(state_tile * k[None, :], axis=1)  # [BLOCK_V]

        # Delta update: state += k^T outer (beta * (v - k_dot_s))
        delta_v = beta * (v_vals - k_dot_s)  # [BLOCK_V]

        # state_tile += outer(delta_v, k)
        state_tile = state_tile + delta_v[:, None] * k[None, :]

        # Compute output: scale * state_tile @ q -> [BLOCK_V]
        # state_tile is [BLOCK_V, K], q is [K]
        out_vals = scale * tl.sum(state_tile * q[None, :], axis=1)  # [BLOCK_V]

        # Store output
        out_base = t * stride_ot + i_hv * stride_oh
        tl.store(output_ptr + out_base + offs_v * stride_ov, out_vals.to(tl.bfloat16), mask=v_mask)

    # Store final state
    ns_base = i_seq * stride_sn + i_hv * stride_sh
    ns_ptrs = new_state_ptr + ns_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    tl.store(ns_ptrs, state_tile, mask=v_mask[:, None])

@triton.jit
def gdn_prefill_chunk_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, output_ptr, new_state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, cu_seqlens_ptr,
    scale,
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vv,
    stride_sn, stride_sh, stride_sv, stride_sk,
    stride_ot, stride_oh, stride_ov,
    stride_at, stride_ah,
    stride_bt, stride_bh,
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    has_state: tl.constexpr,
):
    """
    GDN Prefill kernel with chunked processing for better instruction-level parallelism.
    Grid: (seq_idx, h_v, v_tile)
    """
    i_seq = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_vt = tl.program_id(2)

    i_hq = i_hv // GVA_RATIO

    seq_start = tl.load(cu_seqlens_ptr + i_seq)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1)
    seq_len = seq_end - seq_start

    if seq_len <= 0:
        return

    offs_v = i_vt * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < V
    offs_k = tl.arange(0, K)

    if has_state:
        state_base = i_seq * stride_sn + i_hv * stride_sh
        s_ptrs = state_ptr + state_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
        state_tile = tl.load(s_ptrs, mask=v_mask[:, None], other=0.0)
    else:
        state_tile = tl.zeros([BLOCK_V, K], dtype=tl.float32)

    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    for i in range(seq_len):
        t = seq_start + i

        q_base = t * stride_qt + i_hq * stride_qh
        q = tl.load(q_ptr + q_base + offs_k).to(tl.float32)

        k_base = t * stride_kt + i_hq * stride_kh
        k = tl.load(k_ptr + k_base + offs_k).to(tl.float32)

        v_base = t * stride_vt + i_hv * stride_vh
        v_vals = tl.load(v_ptr + v_base + offs_v * stride_vv, mask=v_mask, other=0.0).to(tl.float32)

        a_val = tl.load(a_ptr + t * stride_at + i_hv * stride_ah).to(tl.float32)
        b_val = tl.load(b_ptr + t * stride_bt + i_hv * stride_bh).to(tl.float32)

        x = a_val + dt_bias_val
        # Numerically stable softplus
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-tl.exp(A_log_val) * softplus_x)
        beta = tl.sigmoid(b_val)

        state_tile = g * state_tile

        k_dot_s = tl.sum(state_tile * k[None, :], axis=1)

        delta_v = beta * (v_vals - k_dot_s)

        state_tile = state_tile + delta_v[:, None] * k[None, :]

        out_vals = scale * tl.sum(state_tile * q[None, :], axis=1)

        out_base = t * stride_ot + i_hv * stride_oh
        tl.store(output_ptr + out_base + offs_v * stride_ov, out_vals.to(tl.bfloat16), mask=v_mask)

    ns_base = i_seq * stride_sn + i_hv * stride_sh
    ns_ptrs = new_state_ptr + ns_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    tl.store(ns_ptrs, state_tile, mask=v_mask[:, None])

def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill with k-last state layout.
    
    Args:
        q: [total_seq_len, H_q, K] bfloat16
        k: [total_seq_len, H_k, K] bfloat16
        v: [total_seq_len, H_v, V] bfloat16
        state: [N, H_v, V, K] float32 or None
        A_log: [H_v] float32
        a: [total_seq_len, H_v] bfloat16
        dt_bias: [H_v] float32
        b: [total_seq_len, H_v] bfloat16
        cu_seqlens: [N+1] int64
        scale: float32 scalar
    
    Returns:
        output: [total_seq_len, H_v, V] bfloat16
        new_state: [N, H_v, V, K] float32
    """
    total_seq_len, H_q, K = q.shape
    H_v = v.shape[1]
    V = v.shape[2]
    num_seqs = cu_seqlens.shape[0] - 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)
    if isinstance(scale, torch.Tensor):
        scale = float(scale)

    GVA_RATIO = H_v // H_q

    output = torch.zeros(total_seq_len, H_v, V, dtype=torch.bfloat16, device=q.device)
    new_state = torch.zeros(num_seqs, H_v, V, K, dtype=torch.float32, device=q.device)

    has_state = state is not None

    # Choose BLOCK_V based on V
    # V=128, K=128 -> BLOCK_V=32 gives 32*128=4096 elements per tile
    BLOCK_V = 32

    num_v_tiles = triton.cdiv(V, BLOCK_V)

    grid = (num_seqs, H_v, num_v_tiles)

    gdn_prefill_kernel[grid](
        q, k, v,
        state if has_state else q,  # dummy pointer if no state
        output, new_state,
        A_log, a, dt_bias, b, cu_seqlens,
        scale,
        # q strides
        q.stride(0), q.stride(1), q.stride(2),
        # k strides
        k.stride(0), k.stride(1), k.stride(2),
        # v strides
        v.stride(0), v.stride(1), v.stride(2),
        # state strides (use new_state strides as dummy if no state)
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        # output strides
        output.stride(0), output.stride(1), output.stride(2),
        # a strides
        a.stride(0), a.stride(1),
        # b strides
        b.stride(0), b.stride(1),
        # dims
        H_q, H_v, K, V, GVA_RATIO, BLOCK_V,
        CHUNK_SIZE=1,  # not used in this kernel
        has_state=has_state,
        num_warps=4,
        num_stages=2,
    )

    return output, new_state