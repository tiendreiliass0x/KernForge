import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_V': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128}, num_warps=8, num_stages=1),
    ],
    key=['V', 'K'],
)
@triton.jit
def gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, new_state_ptr, output_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, scale_ptr,
    stride_sb, stride_sh, stride_sv, stride_sk,
    B, H_q, H_v, K: tl.constexpr, V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_tv = tl.program_id(2)

    gva_ratio = V // K
    i_hq = i_hv // (H_v // H_q)

    offs_k = tl.arange(0, K)
    q = tl.load(q_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)
    k = tl.load(k_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)

    offs_v = i_tv * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < V
    v = tl.load(v_ptr + i_b * H_v * V + i_hv * V + offs_v, mask=v_mask, other=0.0).to(tl.float32)

    x = tl.load(a_ptr + i_b * H_v + i_hv).to(tl.float32) + tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    g = tl.exp(-tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32)) * tl.softplus(x))
    beta = tl.sigmoid(tl.load(b_ptr + i_b * H_v + i_hv).to(tl.float32))
    scale = tl.load(scale_ptr).to(tl.float32)

    ak = g * beta * k
    dk = g * k

    state_offset = i_b * stride_sb + i_hv * stride_sh
    s_ptrs = state_ptr + state_offset + offs_v[:, None] * stride_sv + offs_k[None, :]
    s_mask = v_mask[:, None]

    state_tile = tl.load(s_ptrs, mask=s_mask, other=0.0)

    sk = tl.sum(state_tile * k[None, :], axis=1)

    state_tile = g * state_tile
    state_tile = state_tile - sk[:, None] * ak[None, :]
    state_tile = state_tile + v[:, None] * dk[None, :]

    output_partial = tl.sum(state_tile * q[None, :], axis=1) * scale

    tl.store(s_ptrs, state_tile, mask=s_mask)
    out_ptrs = output_ptr + i_b * H_v * V + i_hv * V + offs_v
    tl.store(out_ptrs, output_partial.to(tl.bfloat16), mask=v_mask)


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, H_q, K = q.shape
    _, H_v, V = v.shape

    output = torch.empty(B, H_v, V, dtype=torch.bfloat16, device=q.device)
    new_state = state.clone()

    grid = lambda meta: (B, H_v, triton.cdiv(V, meta['BLOCK_V']))
    gdn_decode_kernel[grid](
        q, k, v, new_state, new_state, output,
        A_log, a, dt_bias, b, scale,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        B, H_q, H_v, K, V,
    )
    return output, new_state