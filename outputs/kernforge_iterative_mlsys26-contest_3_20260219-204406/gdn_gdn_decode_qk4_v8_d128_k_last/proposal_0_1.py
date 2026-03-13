import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BV": 32}, num_warps=4, num_stages=3),
        triton.Config({"BV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BV": 64}, num_warps=8, num_stages=4),
        triton.Config({"BV": 128}, num_warps=8, num_stages=3),
        triton.Config({"BV": 256}, num_warps=8, num_stages=2),
    ],
    key=["V", "K"],
)
@triton.jit
def _gdn_decode_qk4_v8_d128_k_last_kernel(
    q_ptr, k_ptr, v_ptr,
    state_ptr, out_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    scale_ptr,
    # strides (elements)
    stride_qb, stride_qt, stride_qh, stride_qk,
    stride_kb, stride_kt, stride_kh, stride_kk,
    stride_vb, stride_vt, stride_vh, stride_vk,
    stride_sb, stride_sh, stride_sv, stride_sk,
    stride_ob, stride_ot, stride_oh, stride_ok,
    stride_ab, stride_at, stride_ah,
    stride_bb, stride_bt, stride_bh,
    # sizes
    B: tl.constexpr,
    Hq: tl.constexpr,
    Hv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    # meta
    BV: tl.constexpr,
):
    # Grid: (B, Hv, ceil_div(V, BV))
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_tv = tl.program_id(2)

    # Map v-head -> q/k head (GVA)
    ratio = Hv // Hq
    pid_hq = pid_hv // ratio

    # Offsets
    offs_k = tl.arange(0, K)
    offs_v = pid_tv * BV + tl.arange(0, BV)
    mask_v = offs_v < V

    # Load q,k (bf16 -> f32), T=1 so t=0
    q_ptrs = q_ptr + pid_b * stride_qb + 0 * stride_qt + pid_hq * stride_qh + offs_k * stride_qk
    k_ptrs = k_ptr + pid_b * stride_kb + 0 * stride_kt + pid_hq * stride_kh + offs_k * stride_kk
    q = tl.load(q_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)
    k = tl.load(k_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)

    # Load v tile (bf16 -> f32), T=1
    v_ptrs = v_ptr + pid_b * stride_vb + 0 * stride_vt + pid_hv * stride_vh + offs_v * stride_vk
    v = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

    # Gates:
    # x = a + dt_bias; g = exp(-exp(A_log) * softplus(x)); beta = sigmoid(b)
    # a,b are [B,1,Hv]
    a_val = tl.load(a_ptr + pid_b * stride_ab + 0 * stride_at + pid_hv * stride_ah).to(tl.float32)
    b_val = tl.load(b_ptr + pid_b * stride_bb + 0 * stride_bt + pid_hv * stride_bh).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    A_log = tl.load(A_log_ptr + pid_hv).to(tl.float32)

    x = a_val + dt_bias
    sp = tl.softplus(x)
    g = tl.exp(-tl.exp(A_log) * sp)
    beta = tl.sigmoid(b_val)

    scale = tl.load(scale_ptr).to(tl.float32)

    # State tile pointers: state [B, Hv, V, K] (k-last)
    s_base = state_ptr + pid_b * stride_sb + pid_hv * stride_sh
    s_ptrs = s_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    s_mask = mask_v[:, None] & (offs_k[None, :] < K)

    # Load state tile (f32)
    s = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    # Compute old_v = k @ (g * S_old^T?) reference uses h_state = state.transpose(-1,-2) and k @ h_state
    # For k-last layout state is [V,K] (rows V, cols K). In reference, h_state is [K,V].
    # old_v = k_h @ old_state where old_state is [K,V], equals (old_state^T @ k)^T = ( (g*state)^T @ k )^T
    # For each v-row: dot(state[v,:], k) gives (state @ k) [V]. This equals (old_state^T @ k) with transpose.
    # So compute sk = sum_j s[v,j] * k[j]  (with s already = state_old[v,j])
    # Then old_v[v] = g * sk[v]
    sk = tl.sum(s * k[None, :], axis=1)  # [BV]
    old_v = g * sk  # [BV]

    # new_v = beta*v + (1-beta)*old_v
    new_v = beta * v + (1.0 - beta) * old_v

    # Update state:
    # h_state_new = g*h_state_old - k^T@old_v + k^T@new_v
    # In our [V,K] layout, this becomes:
    # state_new[v,j] = g*state_old[v,j] + (new_v[v] - old_v[v]) * k[j]
    # since outer(k, vec) in [K,V] corresponds to adding vec[v]*k[j] to state_new[v,j] in [V,K].
    delta = (new_v - old_v)[:, None] * k[None, :]  # [BV,K]
    s_new = g * s + delta

    # Output: output[v] = scale * (q @ h_state_new)[v] where h_state_new is [K,V]
    # Equivalent to scale * sum_j q[j] * state_new[v,j]
    out_tile = tl.sum(s_new * q[None, :], axis=1) * scale  # [BV]

    # Store state and output
    tl.store(s_ptrs, s_new, mask=s_mask)
    out_ptrs = out_ptr + pid_b * stride_ob + 0 * stride_ot + pid_hv * stride_oh + offs_v * stride_ok
    tl.store(out_ptrs, out_tile.to(tl.bfloat16), mask=mask_v)


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    # q: [B,1,4,128] bf16
    # k: [B,1,4,128] bf16
    # v: [B,1,8,128] bf16
    # state: [B,8,128,128] f32 (optional)
    # A_log: [8] f32
    # a: [B,1,8] bf16
    # dt_bias: [8] f32
    # b: [B,1,8] bf16
    # scale: f32 scalar (or None/0 -> 1/sqrt(128))
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, Tq, Hq, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, Hv, V = v.shape
    assert Tq == 1 and Tk == 1 and Tv == 1
    assert B == Bk == Bv
    assert Hq == 4 and Hk == 4 and Hv == 8
    assert K == 128 and Kk == 128 and V == 128
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert A_log.dtype == torch.float32 and dt_bias.dtype == torch.float32
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.shape == (B, 1, Hv)
    assert b.shape == (B, 1, Hv)
    assert A_log.shape == (Hv,)
    assert dt_bias.shape == (Hv,)

    if scale is None or (isinstance(scale, (float, int)) and float(scale) == 0.0):
        scale_val = 1.0 / math.sqrt(K)
    else:
        scale_val = float(scale)

    if state is None:
        state = torch.zeros((B, Hv, V, K), device=q.device, dtype=torch.float32)
    else:
        assert state.shape == (B, Hv, V, K)
        assert state.dtype == torch.float32
        state = state.contiguous()

    # output [B,1,Hv,V] bf16
    out = torch.empty((B, 1, Hv, V), device=q.device, dtype=torch.bfloat16)

    scale_t = torch.tensor(scale_val, device=q.device, dtype=torch.float32)

    grid = lambda meta: (B, Hv, triton.cdiv(V, meta["BV"]))

    _gdn_decode_qk4_v8_d128_k_last_kernel[grid](
        q, k, v,
        state, out,
        A_log, a, dt_bias, b,
        scale_t,
        # strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        # sizes
        B=B, Hq=Hq, Hv=Hv, K=K, V=V,
    )
    return out, state