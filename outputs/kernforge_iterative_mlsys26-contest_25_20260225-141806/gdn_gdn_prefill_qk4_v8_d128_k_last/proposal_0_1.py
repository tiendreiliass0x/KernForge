import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gdn_prefill_intra_chunk_kernel(
    # Input pointers
    q_ptr, k_ptr, v_ptr,
    g_ptr, beta_ptr,
    # State pointers
    state_ptr,  # [H_v, V, K] for current sequence
    # Output pointers
    output_ptr,
    # Strides for q [seq, H_q, K]
    stride_qt, stride_qh, stride_qk,
    # Strides for k [seq, H_k, K]
    stride_kt, stride_kh, stride_kk,
    # Strides for v [seq, H_v, V]
    stride_vt, stride_vh, stride_vv,
    # Strides for g [seq, H_v]
    stride_gt, stride_gh,
    # Strides for beta [seq, H_v]
    stride_bt, stride_bh,
    # Strides for state [H_v, V, K]
    stride_sh, stride_sv, stride_sk,
    # Strides for output [seq, H_v, V]
    stride_ot, stride_oh, stride_ov,
    # Sequence info
    seq_start, seq_len,
    # Constants
    H_q: tl.constexpr, H_v: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BV: tl.constexpr,
    scale,
):
    """
    Process one chunk of tokens for one v_head, one v_tile.
    Grid: (num_chunks, H_v, V // BV)
    """
    i_chunk = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_bv = tl.program_id(2)

    # GVA mapping
    gva_ratio = H_v // H_q  # = 2
    i_hq = i_hv // gva_ratio

    chunk_start = seq_start + i_chunk * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, seq_start + seq_len)
    actual_chunk_len = chunk_end - chunk_start

    offs_v = i_bv * BV + tl.arange(0, BV)
    v_mask = offs_v < V
    offs_k = tl.arange(0, K)

    # Load state tile [BV, K]
    state_base = state_ptr + i_hv * stride_sh
    s_ptrs = state_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    s_mask = v_mask[:, None]
    state_tile = tl.load(s_ptrs, mask=s_mask, other=0.0)  # [BV, K], f32

    # Process tokens sequentially within chunk
    for i in range(CHUNK_SIZE):
        t = chunk_start + i
        active = i < actual_chunk_len

        # Load k vector [K]
        k_base = k_ptr + t * stride_kt + i_hq * stride_kh
        k_vec = tl.load(k_base + offs_k, mask=tl.broadcast_to(tl.full([1], active, dtype=tl.int1), [K]), other=0.0).to(tl.float32)

        # Load g scalar
        g_val = tl.load(g_ptr + t * stride_gt + i_hv * stride_gh, mask=active, other=1.0).to(tl.float32)

        # Load beta scalar
        beta_val = tl.load(beta_ptr + t * stride_bt + i_hv * stride_bh, mask=active, other=0.0).to(tl.float32)

        # Load v tile [BV]
        v_base = v_ptr + t * stride_vt + i_hv * stride_vh
        v_vec = tl.load(v_base + offs_v, mask=v_mask & tl.broadcast_to(tl.full([1], active, dtype=tl.int1), [BV]), other=0.0).to(tl.float32)

        # Compute k @ state_tile: [BV] = sum over K of state_tile[bv, k] * k_vec[k]
        sk = tl.sum(state_tile * k_vec[None, :], axis=1)  # [BV]

        # State update:
        # state_new = g * state_old + k^T @ (beta * (v - sk))
        # = g * state_old + k^T @ delta_v
        # where delta_v[bv] = beta * (v[bv] - sk[bv])
        delta_v = beta_val * (v_vec - sk)  # [BV]

        # Apply decay and update
        state_tile = g_val * state_tile + delta_v[:, None] * k_vec[None, :]  # [BV, K]

        # Compute output: o[bv] = scale * sum_k(q[k] * state_new[bv, k])
        # Load q vector
        q_base = q_ptr + t * stride_qt + i_hq * stride_qh
        q_vec = tl.load(q_base + offs_k, mask=tl.broadcast_to(tl.full([1], active, dtype=tl.int1), [K]), other=0.0).to(tl.float32)

        out_val = scale * tl.sum(state_tile * q_vec[None, :], axis=1)  # [BV]

        # Store output
        out_base = output_ptr + t * stride_ot + i_hv * stride_oh
        tl.store(out_base + offs_v,
                 out_val.to(tl.bfloat16),
                 mask=v_mask & tl.broadcast_to(tl.full([1], active, dtype=tl.int1), [BV]))

    # Store updated state
    tl.store(s_ptrs, state_tile, mask=s_mask)