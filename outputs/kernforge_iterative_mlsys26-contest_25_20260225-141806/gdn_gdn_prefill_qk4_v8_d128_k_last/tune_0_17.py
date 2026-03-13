import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def gdn_prefill_kernel(
    # Input pointers
    q_ptr, k_ptr, v_ptr,
    g_ptr, beta_ptr,
    # State pointer (k-last layout [H_v, V, K])
    state_ptr,
    # Output pointer
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
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    scale: tl.constexpr,
):
    """
    Grid: (H_v, V // BV)
    Each program owns one v_head and one BV-tile of the V dimension.
    Processes all tokens sequentially to maintain recurrent state.
    """
    i_hv = tl.program_id(0)
    i_bv = tl.program_id(1)

    # GVA: num_v_heads / num_q_heads = 2, map v_head -> q/k head
    gva_ratio = H_v // H_q
    i_hq = i_hv // gva_ratio

    offs_v = i_bv * BV + tl.arange(0, BV)  # [BV]
    offs_k = tl.arange(0, K)               # [K]
    v_mask = offs_v < V                    # [BV]

    # Load state tile [BV, K] from k-last layout [H_v, V, K]
    state_base = state_ptr + i_hv * stride_sh
    s_ptrs = state_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    s_mask = v_mask[:, None]  # [BV, 1] broadcast to [BV, K]
    state_tile = tl.load(s_ptrs, mask=s_mask, other=0.0)  # [BV, K], f32

    # Process tokens sequentially
    for i in range(seq_len):
        t = seq_start + i

        # Load g scalar (decay gate) for this v_head
        g_raw = tl.load(g_ptr + t * stride_gt + i_hv * stride_gh)
        g_val = tl.cast(g_raw, tl.float32)

        # Load beta scalar (update gate) for this v_head
        beta_raw = tl.load(beta_ptr + t * stride_bt + i_hv * stride_bh)
        beta_val = tl.cast(beta_raw, tl.float32)

        # Load k vector [K] for the corresponding q/k head
        k_base = k_ptr + t * stride_kt + i_hq * stride_kh
        k_raw = tl.load(k_base + offs_k * stride_kk)
        k_vec = tl.cast(k_raw, tl.float32)  # [K]

        # Load v tile [BV] for this v_head
        v_base = v_ptr + t * stride_vt + i_hv * stride_vh
        v_raw = tl.load(v_base + offs_v * stride_vv, mask=v_mask, other=0.0)
        v_vec = tl.cast(v_raw, tl.float32)  # [BV]

        # sk[bv] = state[bv, :] @ k  (k @ state in [BV, K] x [K] -> [BV])
        sk = tl.sum(state_tile * k_vec[None, :], axis=1)  # [BV]

        # Delta rule: new_v = beta * v + (1 - beta) * sk
        # state update = state * g + k^T @ (new_v - sk)
        #              = state * g + k^T @ (beta * (v - sk))
        delta_v = beta_val * (v_vec - sk)  # [BV]

        # State update: state = g * state + outer(delta_v, k)
        state_tile = g_val * state_tile + delta_v[:, None] * k_vec[None, :]  # [BV, K]

        # Compute output: o[bv] = scale * state[bv, :] @ q
        q_base = q_ptr + t * stride_qt + i_hq * stride_qh
        q_raw = tl.load(q_base + offs_k * stride_qk)
        q_vec = tl.cast(q_raw, tl.float32)  # [K]

        out_val = scale * tl.sum(state_tile * q_vec[None, :], axis=1)  # [BV]

        # Store output to output[t, i_hv, offs_v]
        out_base = output_ptr + t * stride_ot + i_hv * stride_oh
        out_bf16 = tl.cast(out_val, tl.bfloat16)
        tl.store(out_base + offs_v * stride_ov, out_bf16, mask=v_mask)

    # Write updated state tile back
    tl.store(s_ptrs, state_tile, mask=s_mask)

@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill using Triton kernel.
    State layout: [N, H_v, V, K] (k-last).
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device

    K_dim = head_size
    V_dim = head_size

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Precompute g and beta on host (numerically stable)
    x = a.float() + dt_bias.float()                        # [T, H_v]
    g = (-A_log.float().exp() * F.softplus(x)).exp()       # [T, H_v]
    beta = torch.sigmoid(b.float())                         # [T, H_v]

    # Ensure contiguous layouts
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    # Allocate outputs
    output = torch.zeros((total_seq_len, num_v_heads, V_dim), dtype=torch.bfloat16, device=device)
    new_state = torch.zeros((num_seqs, num_v_heads, V_dim, K_dim), dtype=torch.float32, device=device)

    # Extract strides
    stride_qt, stride_qh, stride_qk = q.stride()
    stride_kt, stride_kh, stride_kk = k.stride()
    stride_vt, stride_vh, stride_vv = v.stride()
    stride_gt, stride_gh = g.stride()
    stride_bt, stride_bh = beta.stride()
    stride_ot, stride_oh, stride_ov = output.stride()

    # BV=32: 4 tiles of 32 across V=128
    BV = 32

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx])
        seq_end = int(cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        # Build per-sequence state [H_v, V, K]
        if state is not None:
            seq_state = state[seq_idx].clone().float().contiguous()  # [H_v, V, K]
        else:
            seq_state = torch.zeros((num_v_heads, V_dim, K_dim), dtype=torch.float32, device=device)

        stride_sh, stride_sv, stride_sk = seq_state.stride()

        num_bv = (V_dim + BV - 1) // BV
        grid = (num_v_heads, num_bv)

        gdn_prefill_kernel[grid](
            q, k, v,
            g, beta,
            seq_state,
            output,
            stride_qt, stride_qh, stride_qk,
            stride_kt, stride_kh, stride_kk,
            stride_vt, stride_vh, stride_vv,
            stride_gt, stride_gh,
            stride_bt, stride_bh,
            stride_sh, stride_sv, stride_sk,
            stride_ot, stride_oh, stride_ov,
            seq_start, seq_len,
            H_q=num_q_heads,
            H_v=num_v_heads,
            K=K_dim,
            V=V_dim,
            BV=BV,
            scale=scale,
        )

        new_state[seq_idx] = seq_state

    return output, new_state