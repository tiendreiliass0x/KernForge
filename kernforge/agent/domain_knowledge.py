"""
KernForge Domain Intelligence

This is the actual competitive advantage. A generic LLM + "write me a fast kernel"
produces mediocre code. What makes a kernel engineer good is:

1. Pattern recognition: "this workload shape → this tiling strategy"
2. Profiling literacy: "72% memory bandwidth utilization → we're memory bound → reduce traffic"
3. Hardware mental model: "128×128 f32 state = 64KB, fits in B200 SRAM with room to spare"
4. Knowing the tricks: WY decomposition, persistent kernels, swizzled layouts, etc.

This module encodes all of that as structured knowledge the agent can use.
"""

# =============================================================================
# 1. TRITON OPTIMIZATION PATTERNS
#    Concrete, copy-pasteable patterns the agent should know
# =============================================================================

TRITON_PATTERNS = """
## Triton Optimization Patterns for B200

### Pattern 1: Tiled Matrix-Vector Product (for decode kernels)
When computing S @ v where S is [D, D] and v is [D]:
```python
# Tile over rows of S, load v once into SRAM
# Key insight: v is reused across all row-tiles → load once, use many times
@triton.jit
def matvec_tiled(S_ptr, v_ptr, out_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row_id = tl.program_id(0)
    row_offset = row_id * BLOCK_D
    rows = row_offset + tl.arange(0, BLOCK_D)
    mask = rows < D
    
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in range(0, D, BLOCK_D):
        cols = k + tl.arange(0, BLOCK_D)
        col_mask = cols < D
        # Load tile of S: [BLOCK_D, BLOCK_D]
        s_tile = tl.load(S_ptr + rows[:, None] * D + cols[None, :], 
                         mask=mask[:, None] & col_mask[None, :])
        # Load chunk of v: [BLOCK_D]
        v_chunk = tl.load(v_ptr + cols, mask=col_mask)
        # Accumulate: acc += S_tile @ v_chunk
        acc += tl.sum(s_tile * v_chunk[None, :], axis=1)
    
    tl.store(out_ptr + rows, acc, mask=mask)
```

### Pattern 2: Rank-1 State Update (for delta rule)
When computing S = α·S - β·(S@k)·k^T + dt·v·k^T:
```python
# Fuse all three operations in one pass over S tiles
# Key insight: S@k is a reduction over columns, then the update is element-wise
# So we can compute S@k AND update S in the same tile loop

# Step 1: Compute S@k for all rows (one pass)
# Step 2: Update S in-place (second pass, or fused with step 1 if registers allow)
# The outer products v·k^T and (S@k)·k^T share the k vector → load k once
```

### Pattern 3: Persistent Kernel for Sequential Scan
When processing chunks sequentially (prefill inter-chunk state):
```python
# Don't launch one kernel per chunk — use a persistent kernel
# Each thread block handles one (head, batch) pair across ALL chunks
# State stays in registers/shared memory between chunks

@triton.jit
def persistent_scan(state_ptr, ..., num_chunks, CHUNK_SIZE: tl.constexpr):
    head_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # Load initial state into registers
    state = tl.load(...)  # [D, D] tile
    
    for chunk_idx in range(num_chunks):
        # Process chunk — state stays in SRAM
        # ... compute within-chunk, update state ...
    
    # Write final state back to HBM (only once!)
    tl.store(...)
```

### Pattern 4: Vectorized Loads for Contiguous Access
```python
# BAD: Strided access (each thread reads non-adjacent memory)
x = tl.load(ptr + row * stride_row + col)

# GOOD: Contiguous access (threads read adjacent elements)
# Ensure the fastest-varying index maps to thread ID
# For k-last layout [H, V, K]: K is contiguous → tile over V, vectorize K
```

### Pattern 5: Autotune Configuration Space
```python
@triton.autotune(
    configs=[
        # (BLOCK_V, BLOCK_K, num_warps, num_stages)
        # Small batch: fewer warps, more work per warp
        triton.Config({'BV': 32, 'BK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BV': 64, 'BK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BV': 64, 'BK': 64}, num_warps=4, num_stages=3),
        # Large batch: more warps for parallelism
        triton.Config({'BV': 128, 'BK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BV': 64, 'BK': 128}, num_warps=8, num_stages=3),
    ],
    key=['batch_size'],  # retune when batch size changes
)
```

### Pattern 6: Mixed Precision Accumulation
```python
# State is f32, inputs are bf16
# ALWAYS accumulate in f32, cast to bf16 only at the final output

k_f32 = tl.load(k_ptr, ...).to(tl.float32)  # bf16 → f32
v_f32 = tl.load(v_ptr, ...).to(tl.float32)  # bf16 → f32

# All state arithmetic in f32
state = alpha * state  # f32
Sk = tl.dot(state, k_f32)  # f32 accumulation

# Only cast output
output = tl.dot(state, q_f32)  # f32
tl.store(out_ptr, output.to(tl.bfloat16))  # cast at store
```
"""

# =============================================================================
# 1b. INLINE PTX IN TRITON
#     When Triton's abstractions aren't enough, inject PTX directly.
#     Source: fal.ai "Instruction-level control with Inline Elementwise ASM"
# =============================================================================

INLINE_PTX_PATTERNS = """
## Inline PTX Assembly in Triton

Triton provides `tl.inline_asm_elementwise()` to inject PTX instructions without
leaving Python. This is the escape hatch for when Triton's compiler doesn't
generate the optimal instruction. Use it surgically — bulk of kernel stays Triton,
PTX only where it truly matters.

### When to Use Inline PTX
1. **Fast approximate math**: `rcp.approx.ftz.f32` (reciprocal), `rsqrt.approx.f32`
2. **Packed f16x2 operations**: Process two bf16/fp16 values in one instruction
3. **Special conversions**: `cvt.rn.satfinite.e2m1x2.f32` for FP4 quantization
4. **Bit manipulation**: Pack/unpack bits that Triton doesn't expose directly
5. **Fused compare-and-swap**: Single instruction instead of compare + select

### API Reference
```python
(result,) = tl.inline_asm_elementwise(
    asm="rcp.approx.ftz.f32 $0, $1;",  # PTX instruction
    constraints="=r,r",                  # output=r, input=r (32-bit registers)
    args=[input_tensor],                 # Triton tensor arguments
    dtype=[tl.float32],                  # Output dtype(s)
    is_pure=True,                        # No side effects
    pack=1,                              # Elements per invocation
)
```

### Constraint String Format
- `=r`: output 32-bit register
- `r`: input 32-bit register
- Multiple outputs: `"=r,=r,r,r"` (2 outputs, 2 inputs)
- Placeholders in asm: `$0`=first output, `$1`=first input after outputs

### Pack Parameter
- `pack=1`: One element per PTX invocation (for f32)
- `pack=2`: Two elements packed into 32-bit register (for f16/bf16 → use f16x2 instructions)
- `pack=4`: Four elements packed (for int8 or fp8)

### Patterns for GDN Kernels

#### Fast Sigmoid via PTX
```python
# Standard Triton sigmoid: multiple instructions
y = 1.0 / (1.0 + tl.exp(-x))

# Faster: use ex2.approx + rcp.approx (2 instructions instead of ~6)
(y,) = tl.inline_asm_elementwise(
    asm=\"\"\"
    {
        .reg .f32 neg_x, exp_val, one_plus;
        neg.f32 neg_x, $1;
        ex2.approx.ftz.f32 exp_val, neg_x;  // e^(-x) via 2^(-x/ln2)
        add.f32 one_plus, exp_val, 0f3F800000;  // 1.0 + e^(-x)
        rcp.approx.ftz.f32 $0, one_plus;        // 1/(1+e^(-x))
    }
    \"\"\",
    constraints="=r,r",
    args=[x],
    dtype=[tl.float32],
    is_pure=True,
    pack=1,
)
```
Note: approx variants sacrifice ~1 ULP of precision for ~2x speed on transcendentals.
Only use where reference tolerance allows it (check correctness!).

#### Packed BF16 Operations
When processing pairs of bf16 values (e.g., GVA head pairs):
```python
# Process two bf16 values in one instruction cycle
# pack=2 tells Triton to pack two bf16 into one 32-bit register
(result,) = tl.inline_asm_elementwise(
    asm="fma.rn.bf16x2 $0, $1, $2, $3;",  # fused multiply-add on 2 bf16
    constraints="=r,r,r,r",
    args=[a_bf16, b_bf16, c_bf16],
    dtype=[tl.bfloat16],
    is_pure=True,
    pack=2,
)
```

### Important Caveats
- Inline PTX is architecture-specific — test on target GPU
- `approx` variants reduce precision — always verify correctness
- Limited to elementwise semantics (no shared memory, no warp-level control)
- Register constraints must match exactly or you get silent wrong results
- The compiler can't optimize across inline ASM boundaries
"""

# =============================================================================
# 1c. B200 ADVANCED PATTERNS
#     Blackwell-specific techniques from CuTeDSL grouped GEMM patterns.
#     Source: Veitner "Grouped Blockscaled GEMM Kernel" on B200
# =============================================================================

B200_ADVANCED_PATTERNS = """
## B200 Advanced Kernel Patterns

### Pattern: Persistent Kernels with Warp Specialization
On B200, the most performant kernels use warp specialization: different warps
within a thread block perform different roles simultaneously.

Architecture:
```
Thread Block (on one SM):
  ┌─── TMA Warp ──────── MMA Warps ──────── Epilog Warp ───┐
  │  Loads data from    Execute tensor    Writes results     │
  │  HBM → SMEM via    core operations   from SMEM → HBM    │
  │  TMA engine         on SMEM data      via TMA            │
  └─────────────────────────────────────────────────────────┘
  Connected by pipeline barriers (producer → consumer)
```

In Triton, this maps to:
```python
@triton.jit
def persistent_kernel(...):
    warp_id = tl.program_id(0) % num_warps  # conceptual

    # Each block handles MULTIPLE work tiles sequentially (persistent)
    for tile_idx in range(assigned_start, assigned_end):
        # Phase 1: Load (TMA-like async copy)
        data = tl.load(ptr + tile_offsets[tile_idx])

        # Phase 2: Compute (stays in SRAM)
        result = compute(data)

        # Phase 3: Store
        tl.store(out_ptr + tile_offsets[tile_idx], result)
```

The key advantage: data stays in SRAM between tiles. For GDN state scanning,
this means the state matrix is loaded ONCE and updated across all chunks.

### Pattern: TMA (Tensor Memory Accelerator) for Async Bulk Copies
B200's TMA engine performs bulk HBM→SMEM copies asynchronously while the SM
computes on previously loaded data. This is the hardware version of software
pipelining.

In Triton, TMA is partially exposed through `num_stages`:
```python
@triton.autotune(configs=[
    triton.Config({...}, num_stages=3),  # 3 stages = 2 prefetch + 1 compute
])
```
More stages = more data prefetched = better latency hiding, at the cost of
more SMEM used for staging buffers.

For B200 with 256KB SMEM: you can afford 3-4 stages of 64KB state tiles.

### Pattern: Grouped Problem Handling (for Variable-Length Sequences)
From the CuTeDSL grouped GEMM: when processing multiple groups (= sequences)
of different sizes, use a work tile scheduler that:

1. **Linearizes** all tiles across all groups into a single sequence
2. Each SM claims tiles from this sequence (persistent scheduling)
3. When the group changes, **update the tensor descriptors** (pointers, strides)
4. Skip the update when consecutive tiles are in the same group

For GDN prefill with cu_seqlens:
```python
# Precompute: map linear tile index → (sequence_id, chunk_within_seq)
# At runtime:
for tile_idx in range(my_start, my_end):
    seq_id, chunk_id = tile_mapping[tile_idx]

    if seq_id != prev_seq_id:
        # Load new sequence's state, update pointers
        state = load_state(seq_id)
        prev_seq_id = seq_id

    # Process chunk
    state = update_state(state, chunks[seq_id][chunk_id])
```

This avoids the naive approach of launching separate kernels per sequence,
which wastes SM time on short sequences.

### Pattern: Pipeline Staging for Producer-Consumer
The B200 async pipeline pattern:
```
Stage 0: [LOAD tile N  ] [COMPUTE tile N-2] [STORE tile N-4]
Stage 1: [LOAD tile N+1] [COMPUTE tile N-1] [STORE tile N-3]
Stage 2: [LOAD tile N+2] [COMPUTE tile N  ] [STORE tile N-2]
```

In Triton, `num_stages` controls this. But for GDN's recurrent structure,
we need sequential state dependencies — so pipelining applies to the
*independent* dimensions (batch, heads) not the sequential dimension (time).

### Pattern: Tensormap Management for Dynamic Shapes
When kernel handles multiple problem sizes (grouped GEMM, variable-length),
tensormaps (TMA descriptors) must be updated when the problem shape changes.

Key optimization: only update when the group changes, not every tile.
Use a `last_group_idx` tracker and skip the update for consecutive same-group tiles.

### Mapping to GDN Kernels

| B200 Pattern | GDN Decode Application | GDN Prefill Application |
|---|---|---|
| Persistent kernel | Process all (batch, head) pairs, state in SRAM | Inter-chunk state scan, state stays in SRAM |
| TMA async copy | Async state tile loads while computing previous tile | Async chunk loads while computing WY |
| Warp specialization | TMA warp loads state, MMA warps compute, store warp writes | Same, but MMA warps do chunk-parallel attention |
| Grouped scheduling | N/A (batch is uniform) | Variable-length sequences via cu_seqlens |
| Pipeline staging | 2-3 stages of state tiles | 3-4 stages of QKV chunks |
"""

# =============================================================================
# 2. PERFORMANCE ANALYSIS HEURISTICS
#    Rules for interpreting profiling data
# =============================================================================

PROFILING_HEURISTICS = """
## How to Read Kernel Performance Data

### Determining Compute vs Memory Bound
Given:
- achieved_bandwidth_pct: actual HBM bandwidth / peak bandwidth
- achieved_compute_pct: actual FLOPS / peak FLOPS (for the relevant dtype)

Rules:
- If achieved_bandwidth_pct > 70% AND achieved_compute_pct < 30% → MEMORY BOUND
  → Reduce HBM traffic: better tiling, data reuse, compression, fusion
  
- If achieved_compute_pct > 50% AND achieved_bandwidth_pct < 50% → COMPUTE BOUND
  → Use tensor cores (tl.dot), reduce redundant computation, increase arithmetic intensity
  
- If BOTH are low (< 40%) → LATENCY BOUND or LOW OCCUPANCY
  → Check occupancy (registers per thread, shared memory per block)
  → Check for thread divergence or load imbalance
  → Check for excessive synchronization

### B200 Bandwidth Targets
- Peak HBM: 8 TB/s
- Good: > 5.6 TB/s (70%)
- Acceptable: > 4 TB/s (50%)
- Poor: < 3.2 TB/s (40%) — something is wrong

### B200 Compute Targets (BF16 tensor core)
- Peak: 2250 TFLOPS
- Good: > 1125 TFLOPS (50%)
- For memory-bound kernels, compute utilization will naturally be low — that's fine

### Occupancy Analysis
- Registers per thread × threads per block = total registers per SM
- B200: 65536 registers per SM, max 2048 threads per SM
- At 32 registers/thread → 2048 threads → 100% occupancy
- At 64 registers/thread → 1024 threads → 50% occupancy
- At 128 registers/thread → 512 threads → 25% occupancy
- For memory-bound kernels: higher occupancy helps hide latency
- For compute-bound kernels: lower occupancy is acceptable if ALU is saturated

### Memory Traffic Analysis
For GDN decode (single token, per head):
- State read: 128 × 128 × 4B = 64 KB
- State write: 64 KB
- Inputs (q, k, v, gates): ~128 × 5 × 2B ≈ 1.3 KB (negligible)
- Total per head: ~128 KB
- Total per batch (32 heads): ~4 MB
- For batch=64: ~256 MB
- At 8 TB/s: minimum ~32 μs (this is the floor, can't go faster)

For GDN prefill (per sequence, per head, per chunk):
- State: 64 KB read + write
- QKV per chunk: 3 × chunk_size × 128 × 2B per head
- WY matrices: 2 × 128 × chunk_size × 4B
- Intra-chunk attention: chunk_size² × 4B (small for C=64)
"""

# =============================================================================
# 3. COMMON BUG PATTERNS
#    Things LLMs consistently get wrong in Triton kernels
# =============================================================================

COMMON_BUGS = """
## Common LLM Mistakes in Triton Kernels

### Bug 1: Wrong tl.dot dimensions
```python
# WRONG: tl.dot requires both operands to be 2D with compatible shapes
result = tl.dot(a_vec, b_vec)  # Can't dot two 1D vectors!

# RIGHT: reshape to 2D
result = tl.sum(a_vec * b_vec)  # For vector dot product
# OR
result = tl.dot(a_2d, b_2d)  # [M, K] @ [K, N] → [M, N]
```

### Bug 2: Missing mask on boundary tiles
```python
# WRONG: Reads garbage past tensor boundary
tile = tl.load(ptr + offsets)

# RIGHT: Always mask variable dimensions
mask = offsets < actual_size
tile = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### Bug 3: Accumulating in bf16
```python
# WRONG: Catastrophic precision loss
acc = tl.zeros([BLOCK], dtype=tl.bfloat16)  # BAD
acc += some_bf16_value  # Accumulating in bf16

# RIGHT: Always accumulate in f32
acc = tl.zeros([BLOCK], dtype=tl.float32)  # GOOD
acc += some_bf16_value.to(tl.float32)
```

### Bug 4: Wrong stride calculation
```python
# WRONG: Assuming contiguous layout
ptr = base + row * D + col  # Only works if stride_row == D

# RIGHT: Use actual strides from tensor
ptr = base + row * stride_row + col * stride_col
```

### Bug 5: Grid size mismatch
```python
# WRONG: Grid doesn't cover all work
grid = (batch_size, num_heads)  # Missing dimension!

# RIGHT: Account for tiling
grid = (batch_size, num_heads, triton.cdiv(D, BLOCK_D))
```

### Bug 6: Forgetting that triton.autotune wraps the function
```python
# WRONG: Calling tuned_kernel[grid](...) — grid is passed via autotune
@triton.autotune(...)
@triton.jit
def my_kernel(...):
    ...

# The grid lambda is part of autotune, not the launch
# When using autotune, the grid is specified in the Python wrapper, not via [grid]
```

### Bug 7: tl.dot requires multiples of 16 on tensor cores
```python
# WRONG: Arbitrary tile sizes
result = tl.dot(a, b)  # where a is [17, 32] — FAILS on tensor core

# RIGHT: Tile sizes must be multiples of 16 for tl.dot
# Use BLOCK_M, BLOCK_N, BLOCK_K that are multiples of 16
```

### Bug 8: Not handling the GVA head mapping
```python
# WRONG: Using q_head == v_head
q_h = q[batch, head, :]  # head indexes into num_q_heads

# RIGHT: Map v-head to q/k-head
heads_per_group = num_v_heads // num_q_heads  # = 2
qk_head = v_head // heads_per_group
q_h = q[batch, qk_head, :]
```

### Bug 9: K-last vs V-last state layout
```python
# K-LAST layout: state[batch, head, V_dim, K_dim]
# The outer product v @ k^T has shape [V, K] — matches layout!
# state[b, h, :, :] @ k gives a vector indexed by V_dim

# V-LAST layout: state[batch, head, K_dim, V_dim]
# The outer product would need transposing
# MAKE SURE to check which layout the spec uses
```
"""

# =============================================================================
# 4. GDN-SPECIFIC KNOWLEDGE
#    Deep understanding of the Gated Delta Net recurrence
# =============================================================================

GDN_DOMAIN_KNOWLEDGE = """
## Gated Delta Net — Deep Technical Reference

### The Recurrence (Decode, Single Token)
```
Given at time t:
  q_t ∈ R^d (query, bf16)      k_t ∈ R^d (key, bf16)
  v_t ∈ R^d (value, bf16)      S_{t-1} ∈ R^{d×d} (state, f32)
  α ∈ (0,1) (decay, from sigmoid(A_log))
  β ∈ (0,1) (update gate, from sigmoid(b))
  dt > 0 (delta step, from softplus(a + dt_bias))

State update:
  S_t = α · S_{t-1} · (I - β · k_t · k_t^T) + dt · v_t · k_t^T

Expanded:
  S_t = α · S_{t-1} - α · β · (S_{t-1} @ k_t) · k_t^T + dt · v_t · k_t^T

Output:
  o_t = S_t @ q_t · scale
```

### Key Computational Insights

1. **S_{t-1} @ k_t is the expensive part** — it's a matrix-vector product: d² FLOPs
   - Result is a d-dimensional vector: Sk = S @ k
   - This vector is used in the rank-1 subtraction: S -= αβ · Sk · k^T

2. **The update is two rank-1 modifications to S**:
   - Subtract: αβ · (S@k) · k^T  (erase old key association)
   - Add: dt · v · k^T  (write new value association)
   - Both share the k vector — k should be loaded once

3. **S_t @ q_t can reuse the updated state**:
   - Compute S_t first, then multiply by q
   - OR fuse: o = α·(S@q) - αβ·(S@k)·(k^T@q) + dt·(v·(k^T@q))
   - The fused form avoids writing S back for the output computation
   - But you still need to write S for the next timestep

4. **The decay α is per-head, not per-element**:
   - Simple scalar multiply on the entire state matrix
   - This is a massive bandwidth operation (64KB × scalar)
   - Can be fused with the Householder update

### Fused Decode Strategy (Recommended)
```
For each (batch, v_head):
  Load S[b, h] into SRAM tiles  (64 KB)
  Load k, v, q (small, ~384 bytes each)
  Compute gates: α, β, dt (trivial)
  
  For each tile of S (BLOCK_V × D):
    s_tile = load S tile                 # [BV, D] f32
    sk_partial = s_tile @ k              # [BV] — partial dot product
    s_tile = α * s_tile                  # decay
    s_tile -= α * β * sk_partial[:, None] * k[None, :]  # erase
    s_tile += dt * v_tile[:, None] * k[None, :]         # write
    o_partial = s_tile @ q               # [BV] — partial output
    store s_tile back to S               # [BV, D] f32
    accumulate output
  
  Store output (bf16)
```

### Prefill: Chunkwise Parallel Form
For sequences of length L, split into L/C chunks of size C.

Within each chunk, the sequential updates can be expressed as:
  S_c = γ_c · S_{c-1} · P_c + U_c

Where:
  γ_c = ∏_{t in chunk} α_t  (cumulative decay)
  P_c = ∏_{t in chunk} (I - β_t · k_t · k_t^T)  (product of Householder-like matrices)
  U_c = accumulated value contributions within chunk

The WY representation expresses P_c as (I - W · Y^T) where W, Y ∈ R^{d×C}.
This enables computing P_c via two d×C matrices instead of C sequential d×d updates.

Intra-chunk outputs use the "attention-like" form:
  For token t in chunk c:
    o_t = (intra-chunk term using tokens before t in same chunk) + S_{c-1} @ q_t

The intra-chunk term is computed in parallel (like attention within the chunk).
"""

# =============================================================================
# 5. SEARCH SPACE DEFINITION
#    What parameters the agent should actually explore
# =============================================================================

SEARCH_SPACE = """
## Kernel Optimization Search Space

For each kernel, these are the tunable parameters worth exploring:

### Tile Sizes (most impactful)
- BLOCK_V: How many rows of the state to process per thread block
  - Options: 16, 32, 64, 128
  - Tradeoff: larger = more data reuse, but more register pressure
  
- BLOCK_K: How many columns to process per inner loop iteration  
  - Options: 32, 64, 128 (full D)
  - Tradeoff: BLOCK_K=D eliminates inner loop but needs more registers

### Launch Configuration
- num_warps: 4, 8, 16
  - More warps = more parallelism within a block
  - But also more register pressure
  
- num_stages: 1, 2, 3, 4
  - Software pipelining depth for hiding memory latency
  - More stages = more registers for prefetching

### Algorithmic Choices
- Fuse output computation with state update? (saves one pass over S)
- Fuse decay with Householder update? (saves one pass over S)
- Process multiple v-heads per thread block? (if D is small enough)
- Use shared memory explicitly vs let Triton manage it?

### Memory Layout Choices
- State tile ordering in SRAM
- K-contiguous vs V-contiguous tile access pattern
- Padding for bank conflict avoidance

### For Prefill Only
- Chunk size C: 32, 64, 128, 256
  - Tradeoff: larger = more parallelism in intra-chunk, but larger WY matrices
- WY computation strategy: explicit vs implicit
- Inter-chunk scan: sequential vs parallel (for very long sequences)
"""


def get_enhanced_system_prompt(kernel_type: str = "general") -> str:
    """Build the full system prompt with domain knowledge injected."""
    
    base = """You are an expert GPU kernel engineer. You write Triton kernels that WIN benchmarks.

Your optimization methodology:
1. UNDERSTAND the math precisely — trace the reference implementation line by line
2. IDENTIFY the bottleneck — compute bound? memory bound? latency bound?
3. DESIGN the tiling — which data goes in SRAM, what's the access pattern?
4. IMPLEMENT correctly first — match the reference output exactly
5. OPTIMIZE incrementally — one change at a time, measure each

You never guess. You reason from hardware specifications and arithmetic intensity.
"""
    
    sections = [base, TRITON_PATTERNS, INLINE_PTX_PATTERNS, PROFILING_HEURISTICS, COMMON_BUGS]
    
    if kernel_type in ("gdn", "gated_delta_net"):
        sections.append(GDN_DOMAIN_KNOWLEDGE)
        sections.append(B200_ADVANCED_PATTERNS)
    
    sections.append(SEARCH_SPACE)
    
    return "\n\n".join(sections)


def get_analysis_prompt(
    achieved_bandwidth_gb_s: float | None = None,
    achieved_tflops: float | None = None,
    occupancy_pct: float | None = None,
    median_latency_us: float | None = None,
    kernel_type: str = "general",
) -> str:
    """Build a performance analysis prompt with concrete numbers."""
    
    lines = ["## Measured Performance"]
    
    if median_latency_us is not None:
        lines.append(f"- Median latency: {median_latency_us:.1f} μs")
    
    if achieved_bandwidth_gb_s is not None:
        pct = achieved_bandwidth_gb_s / 8000 * 100  # B200 peak
        lines.append(f"- HBM Bandwidth: {achieved_bandwidth_gb_s:.0f} GB/s ({pct:.0f}% of peak)")
        if pct > 70:
            lines.append("  → MEMORY BOUND: reduce HBM traffic")
        elif pct < 30:
            lines.append("  → Low bandwidth utilization: check access patterns")
    
    if achieved_tflops is not None:
        pct_bf16 = achieved_tflops / 2250 * 100  # B200 BF16 peak
        pct_f32 = achieved_tflops / 90 * 100  # B200 FP32 peak
        lines.append(f"- Compute: {achieved_tflops:.1f} TFLOPS ({pct_bf16:.0f}% BF16 peak, {pct_f32:.0f}% FP32 peak)")
        if pct_bf16 > 50:
            lines.append("  → COMPUTE BOUND: optimize arithmetic, use tensor cores")
    
    if occupancy_pct is not None:
        lines.append(f"- Occupancy: {occupancy_pct:.0f}%")
        if occupancy_pct < 25:
            lines.append("  → LOW OCCUPANCY: reduce register pressure or shared memory usage")
    
    return "\n".join(lines)


# =============================================================================
# 6. STRUCTURED MUTATION STRATEGIES
#    Instead of "make it faster", give the agent specific recipes
# =============================================================================

MUTATION_STRATEGIES = [
    {
        "name": "tile_size_sweep",
        "description": "Try different BLOCK_V and BLOCK_K tile sizes",
        "when": "Always try this first — it's the highest-impact knob",
        "instruction": "Change BLOCK_V from {current} to {target}. Keep everything else the same.",
        "variants": [
            {"BV": 16, "BK": 128},
            {"BV": 32, "BK": 128},
            {"BV": 64, "BK": 128},
            {"BV": 64, "BK": 64},
            {"BV": 128, "BK": 128},
        ],
    },
    {
        "name": "fuse_operations", 
        "description": "Fuse decay + householder + value write into one pass over state tiles",
        "when": "When there are multiple passes over the state matrix",
        "instruction": "Combine the α·S scaling, the S@k reduction, the rank-1 subtract, and the rank-1 add into a single tiled loop over S.",
    },
    {
        "name": "vectorize_loads",
        "description": "Ensure contiguous memory access patterns for coalesced loads",
        "when": "When bandwidth utilization is below 60%",
        "instruction": "Reorder the tile loop so the innermost dimension is the contiguous (K) dimension. Use tl.load with a contiguous offset pattern.",
    },
    {
        "name": "fuse_output",
        "description": "Compute output o = S @ q during the same pass that updates S",
        "when": "When there's a separate output computation pass",
        "instruction": "After updating each S tile, immediately compute the partial output tile @ q and accumulate. This avoids reading S a second time.",
    },
    {
        "name": "increase_warps",
        "description": "Use more warps per block for better memory latency hiding",
        "when": "Memory bound with low occupancy",
        "instruction": "Increase num_warps from {current} to {target}. This increases parallelism within each block, helping hide memory latency.",
    },
    {
        "name": "software_pipelining",
        "description": "Increase num_stages for prefetching",
        "when": "Memory bound with only 1-2 pipeline stages",
        "instruction": "Increase num_stages from {current} to {target}. This allows Triton to overlap memory loads with computation.",
    },
    {
        "name": "multi_head_per_block",
        "description": "Process 2 or 4 v-heads in a single thread block",
        "when": "When per-head work is too small to saturate an SM",
        "instruction": "Modify the grid to assign multiple v-heads per block. Load multiple state matrices and process them with shared k/q vectors (GVA heads share q/k).",
    },
    {
        "name": "explicit_shared_memory",
        "description": "Manually manage shared memory instead of relying on Triton's allocator",
        "when": "When register spilling is suspected or SRAM utilization is low",
        "instruction": "Use tl.extra.cuda.shared to explicitly allocate shared memory for the state tile. Load from HBM → shared, compute from shared.",
    },
]


def get_mutation_strategy(generation: int, eval_status: str, 
                          is_memory_bound: bool | None = None) -> dict:
    """Select the most appropriate mutation strategy for this generation."""
    
    if eval_status in ("compile_error", "runtime_error", "incorrect"):
        return {
            "name": "fix_error",
            "instruction": "Fix the error. Do not optimize. Just make it correct.",
        }
    
    # Early generations: try basic tile sizes
    if generation < 5:
        idx = generation % len(MUTATION_STRATEGIES[:3])
        return MUTATION_STRATEGIES[idx]
    
    # Mid generations: apply bottleneck-specific strategies
    if is_memory_bound is True:
        candidates = [s for s in MUTATION_STRATEGIES 
                      if s["name"] in ("fuse_operations", "fuse_output", "vectorize_loads", "software_pipelining")]
    elif is_memory_bound is False:
        candidates = [s for s in MUTATION_STRATEGIES
                      if s["name"] in ("tile_size_sweep", "multi_head_per_block")]
    else:
        candidates = MUTATION_STRATEGIES
    
    idx = generation % len(candidates)
    return candidates[idx]
