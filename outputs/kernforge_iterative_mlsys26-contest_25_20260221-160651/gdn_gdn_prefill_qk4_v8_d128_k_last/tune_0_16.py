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
    # Strides for state input [N, H_v, V, K]
    stride_sn, stride_sh, stride_sv, stride_sk,
    # Strides for new_state output [N, H_v, V, K]
    stride_nsn, stride_nsh, stride_nsv, stride_nsk,
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

    # GVA mapping: map v_head index to q_head index
    i_hq = i_hv // GVA_RATIO

    # Get sequence bounds - use int64 for pointer arithmetic
    seq_start = tl.load(cu_seqlens_ptr + i_seq).to(tl.int64)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1).to(tl.int64)
    seq_len = (seq_end - seq_start).to(tl.int32)

    if seq_len <= 0:
        return

    # V tile offsets - BLOCK_V covers a slice of the V dimension
    offs_v = tl.arange(0, BLOCK_V) + i_vt * BLOCK_V
    offs_k = tl.arange(0, K)
    v_mask = offs_v < V

    # Cast indices to int64 for safe pointer arithmetic
    i_seq_64 = i_seq.to(tl.int64)
    i_hv_64 = i_hv.to(tl.int64)
    i_hq_64 = i_hq.to(tl.int64)

    # Load initial state tile [BLOCK_V, K] from state in [N, H_v, V, K] layout
    if has_state:
        s_ptrs = (state_ptr
                  + i_seq_64 * stride_sn
                  + i_hv_64 * stride_sh
                  + offs_v[:, None] * stride_sv
                  + offs_k[None, :] * stride_sk)
        state_tile = tl.load(s_ptrs, mask=v_mask[:, None], other=0.0).to(tl.float32)
    else:
        state_tile = tl.zeros([BLOCK_V, K], dtype=tl.float32)

    # Load A_log, dt_bias once per head (constant across tokens)
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    # Precompute exp(A_log) for efficiency
    exp_A_log = tl.exp(A_log_val)

    # Process tokens sequentially (required for recurrence)
    for i in range(seq_len):
        t = seq_start + i

        # Load q[t, i_hq, :] -> [K]
        q_vec = tl.load(q_ptr + t * stride_qt + i_hq_64 * stride_qh + offs_k * stride_qk).to(tl.float32)

        # Load k[t, i_hq, :] -> [K]
        k_vec = tl.load(k_ptr + t * stride_kt + i_hq_64 * stride_kh + offs_k * stride_kk).to(tl.float32)

        # Load v[t, i_hv, offs_v] -> [BLOCK_V]
        v_vals = tl.load(v_ptr + t * stride_vt + i_hv_64 * stride_vh + offs_v * stride_vv,
                         mask=v_mask, other=0.0).to(tl.float32)

        # Load gate inputs a[t, i_hv] and b[t, i_hv]
        a_val = tl.load(a_ptr + t * stride_at + i_hv_64 * stride_ah).to(tl.float32)
        b_val = tl.load(b_ptr + t * stride_bt + i_hv_64 * stride_bh).to(tl.float32)

        # Compute g = exp(-exp(A_log) * softplus(a + dt_bias))
        x = a_val + dt_bias_val
        # Numerically stable softplus
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-exp_A_log * softplus_x)

        # Compute beta = sigmoid(b)
        beta = tl.sigmoid(b_val)

        # Apply decay: state = g * state
        state_tile = g * state_tile

        # Compute state @ k -> [BLOCK_V]: old_v[v] = sum_k(state[v,k] * k[k])
        old_v = tl.sum(state_tile * k_vec[None, :], axis=1)  # [BLOCK_V]

        # Delta update: state += outer(beta*(v - old_v), k)
        delta_v = beta * (v_vals - old_v)  # [BLOCK_V]
        state_tile = state_tile + delta_v[:, None] * k_vec[None, :]

        # Compute output: scale * state @ q -> [BLOCK_V]
        out_vals = scale * tl.sum(state_tile * q_vec[None, :], axis=1)  # [BLOCK_V]

        # Store output[t, i_hv, offs_v]
        tl.store(output_ptr + t * stride_ot + i_hv_64 * stride_oh + offs_v * stride_ov,
                 out_vals.to(tl.bfloat16), mask=v_mask)

    # Store final state tile to new_state
    ns_ptrs = (new_state_ptr
               + i_seq_64 * stride_nsn
               + i_hv_64 * stride_nsh
               + offs_v[:, None] * stride_nsv
               + offs_k[None, :] * stride_nsk)
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

    # Handle scale
    if scale is None:
        scale_f = 1.0 / math.sqrt(K)
    elif isinstance(scale, torch.Tensor):
        scale_f = float(scale.detach().cpu())
        if scale_f == 0.0:
            scale_f = 1.0 / math.sqrt(K)
    else:
        scale_f = float(scale)
        if scale_f == 0.0:
            scale_f = 1.0 / math.sqrt(K)

    GVA_RATIO = H_v // H_q

    output = torch.zeros(total_seq_len, H_v, V, dtype=torch.bfloat16, device=q.device)
    new_state = torch.zeros(num_seqs, H_v, V, K, dtype=torch.float32, device=q.device)

    has_state = state is not None

    if num_seqs == 0 or total_seq_len == 0:
        return output, new_state

    # BLOCK_V=8: state tile is 8x128=1024 float32 = 4KB, lower register pressure
    BLOCK_V = 8

    num_v_tiles = triton.cdiv(V, BLOCK_V)
    grid = (num_seqs, H_v, num_v_tiles)

    # State input strides
    if has_state:
        s_stride_n = state.stride(0)
        s_stride_h = state.stride(1)
        s_stride_v = state.stride(2)
        s_stride_k = state.stride(3)
        state_tensor = state
    else:
        # Dummy strides (won't be read since has_state=False)
        s_stride_n = new_state.stride(0)
        s_stride_h = new_state.stride(1)
        s_stride_v = new_state.stride(2)
        s_stride_k = new_state.stride(3)
        state_tensor = new_state

    gdn_prefill_kernel[grid](
        q, k, v,
        state_tensor,
        output, new_state,
        A_log, a, dt_bias, b, cu_seqlens,
        scale_f,
        # q strides
        q.stride(0), q.stride(1), q.stride(2),
        # k strides
        k.stride(0), k.stride(1), k.stride(2),
        # v strides
        v.stride(0), v.stride(1), v.stride(2),
        # state input strides
        s_stride_n, s_stride_h, s_stride_v, s_stride_k,
        # new_state output strides
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        # output strides
        output.stride(0), output.stride(1), output.stride(2),
        # a strides
        a.stride(0), a.stride(1),
        # b strides
        b.stride(0), b.stride(1),
        # dims
        H_q, H_v, K, V, GVA_RATIO, BLOCK_V,
        has_state=has_state,
        num_warps=2,
        num_stages=1,
    )

    return output, new_state