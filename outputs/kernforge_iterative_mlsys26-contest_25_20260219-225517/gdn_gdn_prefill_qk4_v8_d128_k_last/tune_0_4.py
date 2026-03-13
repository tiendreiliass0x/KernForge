import math
import torch
import triton
import triton.language as tl


@triton.jit
def gdn_prefill_kernel(
    # Input pointers
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    # Output pointers
    output_ptr, new_state_ptr,
    # Sequence info
    cu_seqlens_ptr,
    # Scale
    scale,
    # Strides for q [total_seq_len, H_q, K]
    stride_qt, stride_qh, stride_qk,
    # Strides for k [total_seq_len, H_k, K]
    stride_kt, stride_kh, stride_kk,
    # Strides for v [total_seq_len, H_v, V]
    stride_vt, stride_vh, stride_vv,
    # Strides for state [N, H_v, V, K]
    stride_sn, stride_sh, stride_sv, stride_sk,
    # Strides for a [total_seq_len, H_v]
    stride_at, stride_ah,
    # Strides for b [total_seq_len, H_v]
    stride_bt, stride_bh,
    # Strides for output [total_seq_len, H_v, V]
    stride_ot, stride_oh, stride_ov,
    # Dimensions
    H_q: tl.constexpr, H_v: tl.constexpr, K: tl.constexpr,
    BV: tl.constexpr,
    HAS_STATE: tl.constexpr,
):
    """
    Grid: (num_seqs, H_v, cdiv(V, BV))
    Each block handles one sequence, one v_head, one tile of V dimension.
    Iterates sequentially over tokens in the sequence.
    """
    i_seq = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_tv = tl.program_id(2)

    # GVA mapping: v_head -> q/k head
    gva_ratio = H_v // H_q  # = 2
    i_hq = i_hv // gva_ratio

    # Get sequence boundaries
    seq_start = tl.load(cu_seqlens_ptr + i_seq)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1)
    seq_len = seq_end - seq_start

    # V tile offsets
    offs_v = i_tv * BV + tl.arange(0, BV)
    v_mask = offs_v < K  # V == K == 128 here

    offs_k = tl.arange(0, K)

    # Load initial state tile [BV, K]
    state_base = i_seq * stride_sn + i_hv * stride_sh
    s_ptrs = new_state_ptr + state_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    sv_mask = v_mask[:, None]  # broadcast over K

    if HAS_STATE:
        # Load from input state
        s_in_ptrs = state_ptr + state_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
        state_tile = tl.load(s_in_ptrs, mask=sv_mask, other=0.0)
    else:
        state_tile = tl.zeros([BV, K], dtype=tl.float32)

    # Load A_log for this v_head
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    # Process tokens sequentially
    for i in range(seq_len):
        t = seq_start + i

        # Load q, k for this token (q/k head)
        q_base = t * stride_qt + i_hq * stride_qh
        k_base = t * stride_kt + i_hq * stride_kh

        q_vec = tl.load(q_ptr + q_base + offs_k).to(tl.float32)  # [K]
        k_vec = tl.load(k_ptr + k_base + offs_k).to(tl.float32)  # [K]

        # Load v for this token (v_head), tile [BV]
        v_base = t * stride_vt + i_hv * stride_vh
        v_vec = tl.load(v_ptr + v_base + offs_v, mask=v_mask, other=0.0).to(tl.float32)  # [BV]

        # Load a, b for this token and v_head
        a_val = tl.load(a_ptr + t * stride_at + i_hv * stride_ah).to(tl.float32)
        b_val = tl.load(b_ptr + t * stride_bt + i_hv * stride_bh).to(tl.float32)

        # Compute g = exp(-exp(A_log) * softplus(a + dt_bias))
        x = a_val + dt_bias_val
        # numerically stable softplus(x) = x + log(1 + exp(-x)) for x>=0, else log(1+exp(x))
        softplus_x = tl.where(x >= 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-tl.exp(A_log_val) * softplus_x)
</old_str_1>

<reasoning_2>
The kernel also needs to use proper strides for loading q, k, v vectors. Currently it uses raw `offs_k` without multiplying by stride. For contiguous tensors this is fine (stride=1), but let me verify the actual correctness issue is the softplus or something else.

Actually, looking again at the error - `speedup=0.0` with `latency_ms=None` means it's running but giving wrong results. Let me look more carefully at the state update math.

Reference:
```python
old_state_HKV = g_H11 * state_HKV  # g * state [H,K,V]
old_v_H1V = matmul(k_H1K, old_state_HKV)  # k @ (g*state) -> [H,1,V]
new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
state_remove = einsum('hkl,hlv->hkv', k.T, old_v)  # k^T @ old_v
state_update = einsum('hkl,hlv->hkv', k.T, new_v)  # k^T @ new_v
state_HKV = old_state_HKV - state_remove + state_update
```

The reference works in `[H,K,V]` layout. The kernel works in `[V,K]` layout (k-last in state means `[V,K]` slice).

In the kernel:
- `state_tile[bv, k]` = state[v=bv, k=k] (V-major, K-minor)
- `sk = sum(state_tile * k_vec[None,:], axis=1)` = sum over K of state[v,k]*k[k] = (state @ k)[v] ✓
- `delta_v = beta * (v_vec - sk)` ✓
- `state_tile += delta_v[:,None] * k_vec[None,:]` = state[v,k] += delta_v[v] * k[k] ✓

This looks correct! So the math is right.

Hmm, let me check if the issue is with the `q_vec` loading. The q tensor has shape `[T, H_q, K]` with H_q=4. The kernel loads `q_ptr + q_base + offs_k` where `offs_k = tl.arange(0, K)`. Since stride_qk=1 (contiguous), this is `q[t, i_hq, 0:K]`. ✓

Wait - I need to check if the issue is with strides for `offs_k`. The load is:
```python
q_vec = tl.load(q_ptr + q_base + offs_k).to(tl.float32)
```
But `offs_k` should be multiplied by `stride_qk`. For a contiguous tensor, stride_qk=1, so it's fine. But let me make it explicit.

Actually, I think the real issue might be something simpler. Let me re-examine the `sv_mask`:

```python
sv_mask = v_mask[:, None]  # shape [BV, 1]
```

When BV=32 and V=K=128, we have 4 tiles (i_tv=0,1,2,3), and offs_v = [0..31], [32..63], [64..95], [96..127]. All < 128, so v_mask is all True. The mask doesn't matter here since V=K=128 and BV=32 divides evenly.

Let me look at the output store more carefully. The output has shape `[T, H_v, V]`. The store is:
```python
out_base = t * stride_ot + i_hv * stride_oh
tl.store(output_ptr + out_base + offs_v, o_vec.to(tl.bfloat16), mask=v_mask)
```

`offs_v = i_tv * BV + tl.arange(0, BV)` - these are the V indices. `stride_ov = 1` for contiguous. So `out_base + offs_v` gives `output[t, i_hv, i_tv*BV : i_tv*BV+BV]`. ✓

I'm stumped. Let me try a completely different approach and check if there's a race condition. Multiple blocks (different i_tv) write to the same output token t at different V positions. That should be fine since they write to non-overlapping memory.

Oh wait! I see it now. The `new_state` is initialized with `torch.zeros` and then the kernel stores to it. But if `HAS_STATE=True`, the input state is read from `state` and the output is written to `new_state`. The strides passed are `new_state.stride(...)`. But when loading the input state, we use `state_ptr + state_base + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk` - using `new_state` strides for the `state` pointer! If `state` and `new_state` have the same shape (which they do), the strides are identical. So this is fine.

Let me look at the `q_base` computation again. For q shape `[T, H_q=4, K=128]`:
- stride_qt = 4*128 = 512
- stride_qh = 128
- stride_qk = 1

`q_base = t * 512 + i_hq * 128`. Then `q_ptr + q_base + offs_k` where offs_k=[0..127]. This gives `q[t, i_hq, :]`. ✓

For k shape `[T, H_k=4, K=128]` - same strides. ✓

For v shape `[T, H_v=8, V=128]`:
- stride_vt = 8*128 = 1024
- stride_vh = 128
- stride_vv = 1

`v_base = t * 1024 + i_hv * 128`. Then `v_ptr + v_base + offs_v`. ✓

Hmm, everything looks correct. Let me try running with BV=128 (full V dimension in one tile) to eliminate tiling issues.

Actually, I just realized: the `sv_mask` for the store is `v_mask[:, None]` which is `[BV, 1]`. In Triton, this broadcasts to `[BV, K]` for the store of `state_tile` which is `[BV, K]`. But does Triton actually broadcast masks? Let me use `(v_mask[:, None] & tl.full([1, K], True, dtype=tl.int1))` or just `v_mask[:, None] * tl.ones([1, K], dtype=tl.int1)`.

Actually in Triton, masks do broadcast. So that should be fine.

Wait - I just realized something. The `tl.store` for new_state at the end uses `s_ptrs` which was computed BEFORE the loop. But `s_ptrs` points to `new_state_ptr`. That's correct - we store the final state to new_state.

Let me try a different approach: use BV=128 (full tile) to simplify and eliminate potential tiling bugs.
</reasoning_2>

<old_str_2>
    # Choose BV: tile size over V dimension
    # V=128, K=128, BV=32 gives 4 tiles per head
    BV = 32

    num_v_tiles = triton.cdiv(V, BV)
    grid = (num_seqs, H_v, num_v_tiles)
</old_str_2>
<new_str_2>
    # Choose BV: tile size over V dimension
    # V=128, K=128, BV=128 covers full V in one tile (no tiling boundary issues)
    BV = 128

    num_v_tiles = triton.cdiv(V, BV)
    grid = (num_seqs, H_v, num_v_tiles)
</old_str_2>

<reasoning_3>
With BV=128, the v_mask is always all-True (offs_v = [0..127] < 128 = K), and sv_mask is all-True. This eliminates any potential masking issue. The state_tile is [128, 128] = [V, K] which fits in SRAM for a B200.

Also, let me make the strides explicit for q, k, v loads to be safe.
</reasoning_3>

<old_str_3>
        # Load q, k for this token (q/k head)
        q_base = t * stride_qt + i_hq * stride_qh
        k_base = t * stride_kt + i_hq * stride_kh

        q_vec = tl.load(q_ptr + q_base + offs_k).to(tl.float32)  # [K]
        k_vec = tl.load(k_ptr + k_base + offs_k).to(tl.float32)  # [K]

        # Load v for this token (v_head), tile [BV]
        v_base = t * stride_vt + i_hv * stride_vh
        v_vec = tl.load(v_ptr + v_base + offs_v, mask=v_mask, other=0.0).to(tl.float32)  # [BV]
</old_str_3>
<new_str_3>
        # Load q, k for this token (q/k head)
        q_base = t * stride_qt + i_hq * stride_qh
        k_base = t * stride_kt + i_hq * stride_kh

        q_vec = tl.load(q_ptr + q_base + offs_k * stride_qk).to(tl.float32)  # [K]
        k_vec = tl.load(k_ptr + k_base + offs_k * stride_kk).to(tl.float32)  # [K]

        # Load v for this token (v_head), tile [BV]
        v_base = t * stride_vt + i_hv * stride_vh
        v_vec = tl.load(v_ptr + v_base + offs_v * stride_vv, mask=v_mask, other=0.0).to(tl.float32)  # [BV]
</old_str_3>

<reasoning_4>
Also fix the output store to use stride_ov explicitly.
</reasoning_4>

<old_str_4>
        # Store output
        out_base = t * stride_ot + i_hv * stride_oh
        tl.store(output_ptr + out_base + offs_v, o_vec.to(tl.bfloat16), mask=v_mask)
</old_str_4>
<new_str_4>
        # Store output
        out_base = t * stride_ot + i_hv * stride_oh
        tl.store(output_ptr + out_base + offs_v * stride_ov, o_vec.to(tl.bfloat16), mask=v_mask)
</old_str_4>
        # beta = sigmoid(b)
        beta = tl.sigmoid(b_val)

        # State update:
        # old_state = g * state
        # sk = old_state @ k  (for each v row: sum over K)  -> [BV]
        # new_v = beta * v + (1-beta) * sk
        # state_new = old_state - k^T outer sk + k^T outer new_v
        #           = old_state + k^T outer (new_v - sk)
        #           = g*state + k^T outer (beta*(v - sk))

        # Apply decay
        state_tile = g * state_tile  # [BV, K]

        # Compute sk = state_tile @ k -> [BV]
        sk = tl.sum(state_tile * k_vec[None, :], axis=1)  # [BV]

        # delta_v = beta * (v - sk)
        delta_v = beta * (v_vec - sk)  # [BV]

        # Update state: state += delta_v[:, None] * k[None, :]
        state_tile = state_tile + delta_v[:, None] * k_vec[None, :]  # [BV, K]

        # Compute output: o = scale * state_new @ q -> [BV]
        o_vec = scale * tl.sum(state_tile * q_vec[None, :], axis=1)  # [BV]

        # Store output
        out_base = t * stride_ot + i_hv * stride_oh
        tl.store(output_ptr + out_base + offs_v, o_vec.to(tl.bfloat16), mask=v_mask)

    # Store final state
    tl.store(s_ptrs, state_tile, mask=sv_mask)


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill with GVA configuration and k-last state layout.
    
    Inputs:
        q: [total_seq_len, H_q, K] bfloat16
        k: [total_seq_len, H_k, K] bfloat16
        v: [total_seq_len, H_v, V] bfloat16
        state: [N, H_v, V, K] float32 or None (k-last layout)
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
    V = v.shape[2]  # == K == 128
    num_seqs = cu_seqlens.shape[0] - 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    device = q.device

    output = torch.empty((total_seq_len, H_v, V), dtype=torch.bfloat16, device=device)
    new_state = torch.zeros((num_seqs, H_v, V, K), dtype=torch.float32, device=device)

    # Choose BV: tile size over V dimension
    # V=128, K=128, BV=32 gives 4 tiles per head
    BV = 32

    num_v_tiles = triton.cdiv(V, BV)
    grid = (num_seqs, H_v, num_v_tiles)

    has_state = state is not None

    # Use a dummy state tensor if state is None (won't be accessed due to HAS_STATE=False)
    state_arg = state if has_state else new_state  # dummy, won't be read

    gdn_prefill_kernel[grid](
        q, k, v, state_arg,
        A_log, a, dt_bias, b,
        output, new_state,
        cu_seqlens,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        H_q, H_v, K,
        BV,
        has_state,
    )

    return output, new_state