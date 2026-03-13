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
    cu_seqlens_ptr, scale_val,
    # Output pointers
    output_ptr, new_state_ptr,
    # Strides for q [total_seq_len, num_q_heads, head_size]
    stride_q_t, stride_q_h, stride_q_d,
    # Strides for k [total_seq_len, num_k_heads, head_size]
    stride_k_t, stride_k_h, stride_k_d,
    # Strides for v [total_seq_len, num_v_heads, head_size]
    stride_v_t, stride_v_h, stride_v_d,
    # Strides for state [num_seqs, num_v_heads, head_size, head_size] (k-last: [N,H,V,K])
    stride_s_n, stride_s_h, stride_s_v, stride_s_k,
    # Strides for a [total_seq_len, num_v_heads]
    stride_a_t, stride_a_h,
    # Strides for b [total_seq_len, num_v_heads]
    stride_b_t, stride_b_h,
    # Strides for output [total_seq_len, num_v_heads, head_size]
    stride_o_t, stride_o_h, stride_o_d,
    # Dimensions
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_size: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BK: tl.constexpr,
):
    """
    GDN prefill kernel - processes one (seq, v_head) pair per program.
    Uses chunkwise processing for efficiency.
    Grid: (num_seqs, num_v_heads)
    """
    i_seq = tl.program_id(0)
    i_hv = tl.program_id(1)
    
    # GVA mapping: v_head -> q/k head
    gva_ratio = num_v_heads // num_q_heads  # = 2
    i_hq = i_hv // gva_ratio
    
    # Get sequence bounds
    seq_start = tl.load(cu_seqlens_ptr + i_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    
    if seq_len <= 0:
        return
    
    # Load A_log and dt_bias for this v_head
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    
    # Initialize state from input (k-last layout: [N, H_v, V, K])
    # State shape: [head_size(V), head_size(K)] = [128, 128]
    offs_v = tl.arange(0, head_size)  # V dimension (rows)
    offs_k = tl.arange(0, BK)         # K dimension (cols), BK=128
    
    state_base = i_seq * stride_s_n + i_hv * stride_s_h
    s_ptrs = state_ptr + state_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    
    # Load state: [V, K] = [128, 128]
    if state_ptr != output_ptr:  # state is provided
        state = tl.load(s_ptrs)  # [head_size, BK] = [128, 128]
    else:
        state = tl.zeros([head_size, BK], dtype=tl.float32)
    
    # Process tokens sequentially
    num_chunks = tl.cdiv(seq_len, CHUNK_SIZE)
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, seq_len)
        
        for i in range(CHUNK_SIZE):
            t_local = chunk_start + i
            if t_local >= seq_len:
                break
            t = seq_start + t_local
            
            # Load q, k for this token
            q_base = t * stride_q_t + i_hq * stride_q_h
            k_base = t * stride_k_t + i_hq * stride_k_h
            v_base = t * stride_v_t + i_hv * stride_v_h
            
            q_vec = tl.load(q_ptr + q_base + offs_k).to(tl.float32)  # [K]
            k_vec = tl.load(k_ptr + k_base + offs_k).to(tl.float32)  # [K]
            v_vec = tl.load(v_ptr + v_base + offs_v).to(tl.float32)  # [V]
            
            # Load a and b
            a_val = tl.load(a_ptr + t * stride_a_t + i_hv * stride_a_h).to(tl.float32)
            b_val = tl.load(b_ptr + t * stride_b_t + i_hv * stride_b_h).to(tl.float32)
            
            # Compute g and beta
            x = a_val + dt_bias_val
            # softplus(x) = log(1 + exp(x))
            softplus_x = tl.log(1.0 + tl.exp(x))
            g = tl.exp(-tl.exp(A_log_val) * softplus_x)
            beta = tl.sigmoid(b_val)
            
            # State update:
            # old_state = g * state  [V, K]
            # k_state = old_state @ k  (sum over K dim) -> [V]
            # new_v = beta * v + (1 - beta) * k_state  [V]
            # state = old_state + k^T @ (new_v - k_state)  [V,K]
            # = old_state + (new_v - k_state)[:, None] * k[None, :]
            # = old_state + beta * (v - k_state)[:, None] * k[None, :]
            
            # Decay state
            state = g * state  # [V, K]
            
            # Compute k @ state (dot product for each V row)
            # k_state[v] = sum_k state[v, k] * k_vec[k]
            k_state = tl.sum(state * k_vec[None, :], axis=1)  # [V]
            
            # Delta: beta * (v - k_state)
            delta = beta * (v_vec - k_state)  # [V]
            
            # Update state: state += delta[:, None] * k_vec[None, :]
            state = state + delta[:, None] * k_vec[None, :]  # [V, K]
            
            # Compute output: scale * q @ state^T
            # output[v] = scale * sum_k q[k] * state[v, k]
            out_vec = scale_val * tl.sum(state * q_vec[None, :], axis=1)  # [V]
            
            # Store output
            out_base = t * stride_o_t + i_hv * stride_o_h
            tl.store(output_ptr + out_base + offs_v, out_vec.to(tl.bfloat16))
    
    # Store final state
    ns_base = i_seq * stride_s_n + i_hv * stride_s_h
    ns_ptrs = new_state_ptr + ns_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    tl.store(ns_ptrs, state)

@triton.jit
def gdn_prefill_kernel_v2(
    # Input pointers
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    cu_seqlens_ptr,
    # Output pointers
    output_ptr, new_state_ptr,
    # Strides for q [total_seq_len, num_q_heads, head_size]
    stride_q_t, stride_q_h, stride_q_d,
    # Strides for k [total_seq_len, num_k_heads, head_size]
    stride_k_t, stride_k_h, stride_k_d,
    # Strides for v [total_seq_len, num_v_heads, head_size]
    stride_v_t, stride_v_h, stride_v_d,
    # Strides for state [num_seqs, num_v_heads, head_size, head_size] (k-last: [N,H,V,K])
    stride_s_n, stride_s_h, stride_s_v, stride_s_k,
    # Strides for a [total_seq_len, num_v_heads]
    stride_a_t, stride_a_h,
    # Strides for b [total_seq_len, num_v_heads]
    stride_b_t, stride_b_h,
    # Strides for output [total_seq_len, num_v_heads, head_size]
    stride_o_t, stride_o_h, stride_o_d,
    # Scale
    scale_val: tl.constexpr,
    # Dimensions
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_size: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    """
    GDN prefill kernel with tiled V dimension.
    Grid: (num_seqs, num_v_heads, num_v_tiles)
    Each block handles a BV-sized tile of the V dimension.
    """
    i_seq = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_tv = tl.program_id(2)
    
    # GVA mapping
    gva_ratio = num_v_heads // num_q_heads
    i_hq = i_hv // gva_ratio
    
    # Get sequence bounds
    seq_start = tl.load(cu_seqlens_ptr + i_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    
    if seq_len <= 0:
        return
    
    # Load A_log and dt_bias for this v_head
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    
    # V tile offsets
    offs_v = i_tv * BV + tl.arange(0, BV)
    v_mask = offs_v < head_size
    offs_k = tl.arange(0, BK)
    
    # Load state tile: [BV, BK]
    state_base = i_seq * stride_s_n + i_hv * stride_s_h
    s_ptrs = state_ptr + state_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    
    if state_ptr != new_state_ptr:
        state_tile = tl.load(s_ptrs, mask=v_mask[:, None], other=0.0)
    else:
        state_tile = tl.zeros([BV, BK], dtype=tl.float32)
    
    # Process tokens sequentially
    for t_local in range(seq_len):
        t = seq_start + t_local
        
        # Load k (full K dimension)
        k_base = t * stride_k_t + i_hq * stride_k_h
        k_vec = tl.load(k_ptr + k_base + offs_k).to(tl.float32)  # [BK]
        
        # Load v (BV slice)
        v_base = t * stride_v_t + i_hv * stride_v_h
        v_vec = tl.load(v_ptr + v_base + offs_v, mask=v_mask, other=0.0).to(tl.float32)  # [BV]
        
        # Load a and b for this v_head
        a_val = tl.load(a_ptr + t * stride_a_t + i_hv * stride_a_h).to(tl.float32)
        b_val = tl.load(b_ptr + t * stride_b_t + i_hv * stride_b_h).to(tl.float32)
        
        # Compute g and beta
        x = a_val + dt_bias_val
        softplus_x = tl.log(1.0 + tl.exp(x))
        g = tl.exp(-tl.exp(A_log_val) * softplus_x)
        beta_val = tl.sigmoid(b_val)
        
        # Decay
        state_tile = g * state_tile  # [BV, BK]
        
        # k_state = state @ k: [BV]
        k_state = tl.sum(state_tile * k_vec[None, :], axis=1)  # [BV]
        
        # Delta update
        delta = beta_val * (v_vec - k_state)  # [BV]
        state_tile = state_tile + delta[:, None] * k_vec[None, :]  # [BV, BK]
        
        # Load q and compute output
        q_base = t * stride_q_t + i_hq * stride_q_h
        q_vec = tl.load(q_ptr + q_base + offs_k).to(tl.float32)  # [BK]
        
        out_vec = tl.sum(state_tile * q_vec[None, :], axis=1)  # [BV]
        
        # Store output (scale applied)
        out_base = t * stride_o_t + i_hv * stride_o_h
        tl.store(output_ptr + out_base + offs_v, 
                 (out_vec * scale_val).to(tl.bfloat16), 
                 mask=v_mask)
    
    # Store final state tile
    ns_base = i_seq * stride_s_n + i_hv * stride_s_h
    ns_ptrs = new_state_ptr + ns_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    tl.store(ns_ptrs, state_tile, mask=v_mask[:, None])

# Chunked version for longer sequences - processes CHUNK_SIZE tokens at once using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'CHUNK_SIZE': 16, 'BV': 128}, num_warps=4, num_stages=2),
        triton.Config({'CHUNK_SIZE': 32, 'BV': 128}, num_warps=4, num_stages=2),
        triton.Config({'CHUNK_SIZE': 64, 'BV': 128}, num_warps=8, num_stages=2),
        triton.Config({'CHUNK_SIZE': 16, 'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'CHUNK_SIZE': 32, 'BV': 64}, num_warps=4, num_stages=2),
    ],
    key=['head_size', 'num_v_heads'],
)
@triton.jit
def gdn_prefill_chunked_kernel(
    # Input pointers
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    cu_seqlens_ptr,
    # Output pointers
    output_ptr, new_state_ptr,
    # Strides
    stride_q_t, stride_q_h,
    stride_k_t, stride_k_h,
    stride_v_t, stride_v_h,
    stride_s_n, stride_s_h, stride_s_v, stride_s_k,
    stride_a_t, stride_a_h,
    stride_b_t, stride_b_h,
    stride_o_t, stride_o_h,
    # Scale
    scale_val,
    # Dimensions
    num_q_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_size: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BV: tl.constexpr,
):
    """
    GDN prefill with chunked processing.
    Within each chunk, we process tokens sequentially but with vectorized operations.
    Grid: (num_seqs * num_v_heads, num_v_tiles)
    """
    i_sv = tl.program_id(0)
    i_tv = tl.program_id(1)
    
    i_seq = i_sv // num_v_heads
    i_hv = i_sv % num_v_heads
    
    # GVA mapping
    gva_ratio = num_v_heads // num_q_heads
    i_hq = i_hv // gva_ratio
    
    # Get sequence bounds
    seq_start = tl.load(cu_seqlens_ptr + i_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + i_seq + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    
    if seq_len <= 0:
        return
    
    # Load A_log and dt_bias
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    
    # V tile
    offs_v = i_tv * BV + tl.arange(0, BV)
    v_mask = offs_v < head_size
    offs_k = tl.arange(0, head_size)
    
    # Load state tile
    state_base = i_seq * stride_s_n + i_hv * stride_s_h
    s_ptrs = state_ptr + state_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    
    if state_ptr != new_state_ptr:
        state_tile = tl.load(s_ptrs, mask=v_mask[:, None], other=0.0)
    else:
        state_tile = tl.zeros([BV, head_size], dtype=tl.float32)
    
    # Process tokens
    for t_local in range(seq_len):
        t = seq_start + t_local
        
        k_base = t * stride_k_t + i_hq * stride_k_h
        k_vec = tl.load(k_ptr + k_base + offs_k).to(tl.float32)
        
        v_base = t * stride_v_t + i_hv * stride_v_h
        v_vec = tl.load(v_ptr + v_base + offs_v, mask=v_mask, other=0.0).to(tl.float32)
        
        a_val = tl.load(a_ptr + t * stride_a_t + i_hv * stride_a_h).to(tl.float32)
        b_val = tl.load(b_ptr + t * stride_b_t + i_hv * stride_b_h).to(tl.float32)
        
        x = a_val + dt_bias_val
        softplus_x = tl.log(1.0 + tl.exp(x))
        g = tl.exp(-tl.exp(A_log_val) * softplus_x)
        beta_val = tl.sigmoid(b_val)
        
        state_tile = g * state_tile
        k_state = tl.sum(state_tile * k_vec[None, :], axis=1)
        delta = beta_val * (v_vec - k_state)
        state_tile = state_tile + delta[:, None] * k_vec[None, :]
        
        q_base = t * stride_q_t + i_hq * stride_q_h
        q_vec = tl.load(q_ptr + q_base + offs_k).to(tl.float32)
        
        out_vec = scale_val * tl.sum(state_tile * q_vec[None, :], axis=1)
        
        out_base = t * stride_o_t + i_hv * stride_o_h
        tl.store(output_ptr + out_base + offs_v,
                 out_vec.to(tl.bfloat16),
                 mask=v_mask)
    
    ns_base = i_seq * stride_s_n + i_hv * stride_s_h
    ns_ptrs = new_state_ptr + ns_base + offs_v[:, None] * stride_s_v + offs_k[None, :] * stride_s_k
    tl.store(ns_ptrs, state_tile, mask=v_mask[:, None])

@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill with Triton kernel.
    
    Inputs:
        q: [total_seq_len, num_q_heads, head_size] bfloat16
        k: [total_seq_len, num_k_heads, head_size] bfloat16
        v: [total_seq_len, num_v_heads, head_size] bfloat16
        state: [num_seqs, num_v_heads, head_size, head_size] float32 or None (k-last layout)
        A_log: [num_v_heads] float32
        a: [total_seq_len, num_v_heads] bfloat16
        dt_bias: [num_v_heads] float32
        b: [total_seq_len, num_v_heads] bfloat16
        cu_seqlens: [num_seqs + 1] int64
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
    
    # Allocate outputs
    output = torch.empty(total_seq_len, num_v_heads, head_size, dtype=torch.bfloat16, device=device)
    new_state = torch.zeros(num_seqs, num_v_heads, head_size, head_size, dtype=torch.float32, device=device)
    
    has_state = state is not None
    if has_state:
        state_in = state.contiguous()
    else:
        state_in = new_state  # dummy pointer, won't be read
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    
    BV = min(128, head_size)
    num_v_tiles = triton.cdiv(head_size, BV)
    
    # Grid: (num_seqs * num_v_heads, num_v_tiles)
    grid = (num_seqs * num_v_heads, num_v_tiles)
    
    gdn_prefill_chunked_kernel[grid](
        q, k, v, state_in,
        A_log, a, dt_bias, b,
        cu_seqlens,
        output, new_state,
        # q strides
        q.stride(0), q.stride(1),
        # k strides
        k.stride(0), k.stride(1),
        # v strides
        v.stride(0), v.stride(1),
        # state strides [N, H_v, V, K]
        state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
        # a strides
        a.stride(0), a.stride(1),
        # b strides
        b.stride(0), b.stride(1),
        # output strides
        output.stride(0), output.stride(1),
        # scale
        scale,
        # dimensions
        num_q_heads, num_v_heads, head_size,
        # has_state flag
        has_state,
    )
    
    return output, new_state