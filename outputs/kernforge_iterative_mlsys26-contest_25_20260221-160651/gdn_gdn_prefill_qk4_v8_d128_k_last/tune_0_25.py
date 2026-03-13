import math
import torch
import triton
import triton.language as tl

@triton.jit
def gdn_prefill_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, output_ptr, new_state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, cu_seqlens_ptr,
    scale,
    # Tensor strides (all int32 from Python, cast inside as needed)
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vv,
    stride_sn, stride_sh, stride_sv, stride_sk,
    stride_nsn, stride_nsh, stride_nsv, stride_nsk,
    stride_ot, stride_oh, stride_ov,
    stride_at, stride_ah,
    stride_bt, stride_bh,
    # Constexpr dims
    GVA_RATIO: tl.constexpr,
    K: tl.constexpr,
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

    # GVA mapping: map v_head index to q/k head index
    i_hq = i_hv // GVA_RATIO

    # Get sequence bounds - cu_seqlens is int64
    seq_start = tl.load(cu_seqlens_ptr + i_seq).to(tl.int64)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1).to(tl.int64)
    seq_len = seq_end - seq_start

    if seq_len <= 0:
        return

    # Convert program IDs to int64 for pointer arithmetic
    i_seq_64 = i_seq.to(tl.int64)
    i_hv_64 = i_hv.to(tl.int64)
    i_hq_64 = i_hq.to(tl.int64)

    # Tile offsets
    offs_v = tl.arange(0, BLOCK_V) + i_vt * BLOCK_V  # [BLOCK_V]
    offs_k = tl.arange(0, K)                           # [K]
    offs_v_64 = offs_v.to(tl.int64)
    offs_k_64 = offs_k.to(tl.int64)

    # Precompute static head offsets (int64)
    q_head_off = i_hq_64 * stride_qh.to(tl.int64)
    k_head_off = i_hq_64 * stride_kh.to(tl.int64)
    v_head_off = i_hv_64 * stride_vh.to(tl.int64)
    a_head_off = i_hv_64 * stride_ah.to(tl.int64)
    b_head_off = i_hv_64 * stride_bh.to(tl.int64)
    out_head_off = i_hv_64 * stride_oh.to(tl.int64)

    # Precompute static k/v index offsets
    q_k_off = offs_k_64 * stride_qk.to(tl.int64)   # [K]
    k_k_off = offs_k_64 * stride_kk.to(tl.int64)   # [K]
    v_v_off = offs_v_64 * stride_vv.to(tl.int64)   # [BLOCK_V]
    out_v_off = offs_v_64 * stride_ov.to(tl.int64) # [BLOCK_V]

    # Load initial state tile [BLOCK_V, K] from state [N, H_v, V, K]
    if has_state:
        s_base = (state_ptr
                  + i_seq_64 * stride_sn.to(tl.int64)
                  + i_hv_64 * stride_sh.to(tl.int64))
        s_ptrs = (s_base
                  + offs_v_64[:, None] * stride_sv.to(tl.int64)
                  + offs_k_64[None, :] * stride_sk.to(tl.int64))
        state_tile = tl.load(s_ptrs).to(tl.float32)
    else:
        state_tile = tl.zeros([BLOCK_V, K], dtype=tl.float32)

    # Load per-head constants
    A_log_val = tl.load(A_log_ptr + i_hv_64).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv_64).to(tl.float32)
    exp_A_log = tl.exp(A_log_val)

    # Process tokens sequentially
    for i in range(seq_len):
        t = seq_start + i  # int64 token index

        t_qt = t * stride_qt.to(tl.int64)
        t_kt = t * stride_kt.to(tl.int64)
        t_vt = t * stride_vt.to(tl.int64)
        t_at = t * stride_at.to(tl.int64)
        t_bt = t * stride_bt.to(tl.int64)
        t_ot = t * stride_ot.to(tl.int64)

        # Load q[t, i_hq, :] -> [K]
        q_vec = tl.load(q_ptr + t_qt + q_head_off + q_k_off).to(tl.float32)

        # Load k[t, i_hq, :] -> [K]
        k_vec = tl.load(k_ptr + t_kt + k_head_off + k_k_off).to(tl.float32)

        # Load v[t, i_hv, offs_v] -> [BLOCK_V]
        v_vals = tl.load(v_ptr + t_vt + v_head_off + v_v_off).to(tl.float32)

        # Load a[t, i_hv] and b[t, i_hv]
        a_val = tl.load(a_ptr + t_at + a_head_off).to(tl.float32)
        b_val = tl.load(b_ptr + t_bt + b_head_off).to(tl.float32)

        # g = exp(-exp(A_log) * softplus(a + dt_bias))
        x = a_val + dt_bias_val
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-exp_A_log * softplus_x)

        # beta = sigmoid(b)
        beta = tl.sigmoid(b_val)

        # Decay state
        state_tile = g * state_tile

        # old_v = state @ k  -> [BLOCK_V]
        old_v = tl.sum(state_tile * k_vec[None, :], axis=1)

        # Delta update
        delta_v = beta * (v_vals - old_v)  # [BLOCK_V]
        state_tile = state_tile + delta_v[:, None] * k_vec[None, :]

        # Output = scale * state @ q  -> [BLOCK_V]
        out_vals = scale * tl.sum(state_tile * q_vec[None, :], axis=1)

        # Store output[t, i_hv, offs_v]
        tl.store(output_ptr + t_ot + out_head_off + out_v_off,
                 out_vals.to(tl.bfloat16))

    # Store final state [BLOCK_V, K] -> new_state [N, H_v, V, K]
    ns_base = (new_state_ptr
               + i_seq_64 * stride_nsn.to(tl.int64)
               + i_hv_64 * stride_nsh.to(tl.int64))
    ns_ptrs = (ns_base
               + offs_v_64[:, None] * stride_nsv.to(tl.int64)
               + offs_k_64[None, :] * stride_nsk.to(tl.int64))
    tl.store(ns_ptrs, state_tile)

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

    output = torch.empty(total_seq_len, H_v, V, dtype=torch.bfloat16, device=q.device)
    new_state = torch.zeros(num_seqs, H_v, V, K, dtype=torch.float32, device=q.device)

    has_state = state is not None

    if num_seqs == 0 or total_seq_len == 0:
        return output, new_state

    # BLOCK_V=16: tile is 16x128 float32 = 8KB; V=128 -> 8 tiles
    BLOCK_V = 16
    num_v_tiles = V // BLOCK_V  # V=128 is divisible by 16

    # Grid: (num_seqs, H_v, num_v_tiles)
    grid = (num_seqs, H_v, num_v_tiles)

    # State input tensor and strides
    if has_state:
        state_tensor = state
        s_sn = state.stride(0)
        s_sh = state.stride(1)
        s_sv = state.stride(2)
        s_sk = state.stride(3)
    else:
        # Use new_state as dummy (zeros already); kernel won't read it (has_state=False)
        state_tensor = new_state
        s_sn = new_state.stride(0)
        s_sh = new_state.stride(1)
        s_sv = new_state.stride(2)
        s_sk = new_state.stride(3)

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
        s_sn, s_sh, s_sv, s_sk,
        # new_state output strides
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        # output strides
        output.stride(0), output.stride(1), output.stride(2),
        # a strides
        a.stride(0), a.stride(1),
        # b strides
        b.stride(0), b.stride(1),
        # constexpr dims
        GVA_RATIO=GVA_RATIO,
        K=K,
        BLOCK_V=BLOCK_V,
        has_state=has_state,
        num_warps=4,
        num_stages=1,
    )

    return output, new_state