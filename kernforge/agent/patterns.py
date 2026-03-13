"""
Triton/CUDA kernel optimization patterns — the knowledge that makes agents competitive.

This module encodes specific, actionable optimization patterns that an LLM can
recognize and apply. Each pattern has:
- A condition (when to apply it)
- A transformation (what to change in the code)
- Expected impact (how much faster, and why)

These aren't generic tips — they're the patterns that expert kernel engineers
actually use, mapped to things an LLM can detect and act on.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizationPattern:
    """A specific, actionable kernel optimization pattern."""
    name: str
    category: str  # memory, compute, occupancy, algorithmic, numerical
    condition: str  # when to apply (human-readable for LLM)
    transformation: str  # what to change (specific code guidance)
    expected_impact: str  # quantitative estimate
    priority: int  # 1=critical, 5=minor
    code_example: str = ""  # before/after code snippet


# === MEMORY OPTIMIZATION PATTERNS ===

COALESCE_GLOBAL_LOADS = OptimizationPattern(
    name="coalesce_global_loads",
    category="memory",
    condition=(
        "Threads in the same warp load from non-contiguous memory addresses. "
        "Symptom: L2 cache hit rate is low, or memory throughput is far below peak HBM bandwidth. "
        "Common when iterating over the 'wrong' dimension first."
    ),
    transformation=(
        "Ensure the innermost loop dimension corresponds to contiguous memory. "
        "For row-major tensors, threads should load consecutive elements along the last axis. "
        "Use tl.load with a pointer pattern where thread_id indexes the fastest-varying dimension. "
        "If the natural algorithm iterates column-first, consider transposing the input or "
        "restructuring the tile shape."
    ),
    expected_impact="2-8x improvement on memory-bound kernels",
    priority=1,
    code_example="""
# BAD: threads load from different rows (stride = num_cols between loads)
for i in range(BLOCK_M):
    row_data = tl.load(ptr + i * stride_row + col_offsets)

# GOOD: threads load contiguous within a row
for i in range(BLOCK_M):
    row_data = tl.load(ptr + i * stride_row + tl.arange(0, BLOCK_N))
""",
)

VECTORIZED_LOADS = OptimizationPattern(
    name="vectorized_loads",
    category="memory",
    condition=(
        "Loading individual float32/bfloat16 values instead of packed vectors. "
        "Symptom: memory throughput is well below peak even with coalesced access."
    ),
    transformation=(
        "Use larger BLOCK sizes along the contiguous dimension (128+ elements per thread) "
        "to enable the compiler to emit vectorized loads (LDG.128). "
        "Ensure the pointer is aligned to 16 bytes. "
        "For bf16, BLOCK_K should be at least 128 to get 128-bit loads."
    ),
    expected_impact="1.5-2x on memory-bound kernels",
    priority=2,
)

SHARED_MEMORY_TILING = OptimizationPattern(
    name="shared_memory_tiling",
    category="memory",
    condition=(
        "Data is loaded from HBM multiple times across different loop iterations. "
        "Symptom: the same tensor region is accessed in multiple passes, "
        "or HBM read bytes >> tensor size."
    ),
    transformation=(
        "Tile the computation so that a chunk of data is loaded into shared memory once, "
        "then reused across multiple compute steps. In Triton, this happens automatically "
        "when you load a block into a variable and use it in a loop body. "
        "Key: ensure BLOCK sizes are chosen so the working set fits in SRAM. "
        "B200 has 256KB per SM. A 64x128 float32 tile = 32KB."
    ),
    expected_impact="2-10x for compute-heavy kernels with data reuse",
    priority=1,
)

MINIMIZE_STATE_TRAFFIC = OptimizationPattern(
    name="minimize_state_traffic",
    category="memory",
    condition=(
        "For recurrent kernels (like GDN decode): the state matrix is large "
        "(e.g., 128x128 float32 = 64KB per head) and is read/written every step. "
        "Symptom: kernel is memory-bound despite having significant compute."
    ),
    transformation=(
        "Process the state in tiles that stay in registers/SRAM across the entire "
        "computation. Don't write intermediate state back to HBM. "
        "For GDN decode: load state tile → apply decay → householder update → "
        "delta write → compute output → store state tile. "
        "All 4 operations fused in one pass over the state. "
        "Consider processing multiple sequence tokens per kernel launch if possible."
    ),
    expected_impact="1.5-3x for state-heavy recurrent kernels",
    priority=1,
)

# === COMPUTE OPTIMIZATION PATTERNS ===

USE_TENSOR_CORES = OptimizationPattern(
    name="use_tensor_cores",
    category="compute",
    condition=(
        "Matrix multiplications are done with scalar ops instead of tensor cores. "
        "Symptom: low SM utilization, FLOPS far below BF16 tensor core peak. "
        "tl.dot() uses tensor cores; manual element-wise multiply-add does not."
    ),
    transformation=(
        "Replace element-wise matrix operations with tl.dot(). "
        "Requirements for tl.dot: "
        "  - Both inputs must be 2D tiles "
        "  - Inner dimension must be >= 16 for bf16 "
        "  - Shapes must be multiples of 16 "
        "  - One input is transposed automatically (a @ b.T semantic) "
        "For outer products v @ k^T: reshape to [BLOCK_V, 1] @ [1, BLOCK_K] "
        "or use tl.dot with appropriate reshaping."
    ),
    expected_impact="10-50x for matrix-multiply-heavy kernels",
    priority=1,
)

FUSE_ELEMENTWISE = OptimizationPattern(
    name="fuse_elementwise",
    category="compute",
    condition=(
        "Multiple separate kernels or passes do element-wise operations on the same data. "
        "Common: activation function + scaling + type cast as separate steps."
    ),
    transformation=(
        "Combine all element-wise operations into a single kernel/loop body. "
        "In Triton: just do all operations on the same tile before storing. "
        "sigmoid, softplus, scaling, type casts all fuse naturally. "
        "For GDN: fuse alpha computation, beta computation, state decay, "
        "householder, delta update, and output projection into one kernel."
    ),
    expected_impact="1.2-2x (reduces kernel launch overhead and memory traffic)",
    priority=2,
)

PRECOMPUTE_INVARIANTS = OptimizationPattern(
    name="precompute_invariants",
    category="compute",
    condition=(
        "Values that don't depend on the tile loop variable are recomputed inside the loop. "
        "Common: gate values, scaling factors, broadcast vectors."
    ),
    transformation=(
        "Hoist loop-invariant computations before the tile loop. "
        "For GDN decode: alpha, dt, beta, k_float, v_float, q_float can all be "
        "computed once before iterating over state tiles. "
        "Pre-expand k and q for GVA head mapping before the loop."
    ),
    expected_impact="1.1-1.3x (small but free)",
    priority=3,
)

# === OCCUPANCY PATTERNS ===

REGISTER_PRESSURE = OptimizationPattern(
    name="register_pressure",
    category="occupancy",
    condition=(
        "Kernel uses too many registers per thread, limiting the number of concurrent "
        "warps (occupancy). Symptom: occupancy < 50% in NCU, achieved throughput is low. "
        "Common when tile sizes are too large or too many intermediate values are live."
    ),
    transformation=(
        "Reduce tile sizes (smaller BLOCK_M, BLOCK_N, BLOCK_K). "
        "Recompute values instead of storing them (trade compute for registers). "
        "Use smaller accumulator types where precision allows. "
        "In Triton: use num_warps to control parallelism within a block. "
        "B200: 65536 registers/SM, 2048 max threads/SM. "
        "At 128 registers/thread, max 512 threads = 16 warps = 50% occupancy."
    ),
    expected_impact="1.3-2x when occupancy is the bottleneck",
    priority=2,
)

GRID_SIZING = OptimizationPattern(
    name="grid_sizing",
    category="occupancy",
    condition=(
        "The grid has fewer blocks than SMs, leaving some SMs idle. "
        "Symptom: SM active % is low, or batch_size * num_heads < num_SMs. "
        "B200 has 192 SMs."
    ),
    transformation=(
        "Split work along additional dimensions to increase the grid size. "
        "For GDN decode: if batch=4, heads=32, grid=128 < 192 SMs → 33% idle. "
        "Consider splitting the state matrix itself into tiles that become grid blocks. "
        "E.g., grid = (batch, heads, num_v_tiles) where state is tiled along V dimension."
    ),
    expected_impact="1.2-1.5x when GPU is under-utilized",
    priority=2,
)

# === ALGORITHMIC PATTERNS (specific to linear attention / GDN) ===

CHUNKWISE_PARALLEL = OptimizationPattern(
    name="chunkwise_parallel",
    category="algorithmic",
    condition=(
        "Prefill kernel with seq_len > 1 processes tokens sequentially. "
        "The recurrence S_t = f(S_{t-1}, x_t) prevents parallelism."
    ),
    transformation=(
        "Use the chunkwise parallel form: "
        "1. Split sequence into chunks of size C (64 or 128) "
        "2. Within each chunk, use WY representation to parallelize "
        "   the product of Householder transformations "
        "3. Compute intra-chunk outputs with chunk-local attention "
        "4. Scan inter-chunk states sequentially (but there are few chunks) "
        "5. Add cross-chunk contribution to outputs "
        "This turns O(T) sequential ops into O(T/C) sequential + O(C) parallel."
    ),
    expected_impact="5-20x for long sequences (prefill)",
    priority=1,
)

WY_REPRESENTATION = OptimizationPattern(
    name="wy_representation",
    category="algorithmic",
    condition=(
        "Prefill kernel needs to compute product of rank-1 updates: "
        "∏_t (I - β_t k_t k_t^T) within a chunk."
    ),
    transformation=(
        "Use WY factorization: represent ∏(I - β_t k_t k_t^T) = I - W Y^T "
        "where W, Y ∈ R^{d×C}. "
        "Build W, Y incrementally: "
        "  Y[:, 0] = k_0, W[:, 0] = β_0 k_0 "
        "  For t > 0: "
        "    z = β_t (I - W[:, :t] @ Y[:, :t]^T) @ k_t "
        "    W[:, t] = z, Y[:, t] = k_t "
        "Then apply to state: S @ (I - W Y^T) = S - (S @ W) @ Y^T "
        "This turns O(C) sequential matmuls into O(C) rank-1 updates + one GEMM."
    ),
    expected_impact="2-5x within each chunk",
    priority=1,
)

GVA_HEAD_MAPPING = OptimizationPattern(
    name="gva_head_mapping",
    category="algorithmic",
    condition=(
        "Grouped Value Attention: num_v_heads > num_q_heads. "
        "Multiple v-heads share the same q/k head."
    ),
    transformation=(
        "Don't repeat_interleave q/k — that wastes memory and bandwidth. "
        "Instead, map v-head to q/k head with integer division: qk_head = v_head // ratio. "
        "Load q/k once per q/k head group, compute output for all v-heads in the group. "
        "For GDN with ratio=2: process v-heads in pairs, loading q/k once."
    ),
    expected_impact="1.3-1.5x (eliminates redundant q/k loads)",
    priority=2,
)

STATE_LAYOUT_OPTIMIZATION = OptimizationPattern(
    name="state_layout_optimization",
    category="algorithmic",
    condition=(
        "State matrix access pattern doesn't match the storage layout. "
        "K-last layout: state[B, H, V, K] means rows are V-dim, columns are K-dim."
    ),
    transformation=(
        "Match the tiling to the layout: "
        "  - S @ k: dot product along K (columns) → loads are contiguous ✓ "
        "  - v @ k^T: outer product → v is rows, k is columns → matches K-last ✓ "
        "  - S @ q: dot product along K → contiguous ✓ "
        "K-last layout is favorable for GDN operations. "
        "Tile along V dimension (rows) with BLOCK_V, process full K=128 per tile."
    ),
    expected_impact="1.2-1.5x vs wrong-layout access",
    priority=3,
)


# === NUMERICAL PATTERNS ===

MIXED_PRECISION_ACCUMULATION = OptimizationPattern(
    name="mixed_precision_accumulation",
    category="numerical",
    condition=(
        "Accumulating in bf16 causes numerical drift, especially for "
        "large reductions or state updates over many steps."
    ),
    transformation=(
        "Always accumulate in float32. Cast to bf16 only for the final output store. "
        "In Triton: use .to(tl.float32) before accumulation, .to(tl.bfloat16) at store. "
        "For state: keep in float32 throughout (the spec requires f32 state). "
        "For dot products: tl.dot accumulates in f32 by default."
    ),
    expected_impact="Required for correctness (not optional)",
    priority=1,
)

SOFTPLUS_NUMERICAL_STABILITY = OptimizationPattern(
    name="softplus_stability",
    category="numerical",
    condition="Computing softplus(x) = log(1 + exp(x)) which overflows for large x.",
    transformation=(
        "Use the numerically stable form: "
        "  softplus(x) = x + log(1 + exp(-abs(x)))  if x > 0 "
        "  softplus(x) = log(1 + exp(x))             if x <= 0 "
        "Or simply: softplus(x) = max(x, 0) + log(1 + exp(-abs(x))) "
        "In Triton: tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x))) is often sufficient."
    ),
    expected_impact="Required for correctness with large dt values",
    priority=1,
)


# === PATTERN REGISTRY ===

ALL_PATTERNS = [
    COALESCE_GLOBAL_LOADS,
    VECTORIZED_LOADS,
    SHARED_MEMORY_TILING,
    MINIMIZE_STATE_TRAFFIC,
    USE_TENSOR_CORES,
    FUSE_ELEMENTWISE,
    PRECOMPUTE_INVARIANTS,
    REGISTER_PRESSURE,
    GRID_SIZING,
    CHUNKWISE_PARALLEL,
    WY_REPRESENTATION,
    GVA_HEAD_MAPPING,
    STATE_LAYOUT_OPTIMIZATION,
    MIXED_PRECISION_ACCUMULATION,
    SOFTPLUS_NUMERICAL_STABILITY,
]

PATTERNS_BY_CATEGORY = {}
for p in ALL_PATTERNS:
    PATTERNS_BY_CATEGORY.setdefault(p.category, []).append(p)


def get_relevant_patterns(
    kernel_type: str,
    is_correct: bool,
    is_memory_bound: Optional[bool] = None,
    occupancy_pct: Optional[float] = None,
) -> list[OptimizationPattern]:
    """Select the most relevant patterns given the current situation."""
    patterns = []

    # Always include numerical patterns (correctness first)
    if not is_correct:
        patterns.extend(PATTERNS_BY_CATEGORY["numerical"])
        return sorted(patterns, key=lambda p: p.priority)

    # Kernel-type-specific patterns
    if kernel_type in ("gdn", "linear_attention", "recurrent"):
        patterns.extend([GVA_HEAD_MAPPING, STATE_LAYOUT_OPTIMIZATION, MINIMIZE_STATE_TRAFFIC])
        if "prefill" in kernel_type or kernel_type == "gdn":
            patterns.extend([CHUNKWISE_PARALLEL, WY_REPRESENTATION])

    # Performance-based selection
    if is_memory_bound is True:
        patterns.extend(PATTERNS_BY_CATEGORY["memory"])
    elif is_memory_bound is False:
        patterns.extend(PATTERNS_BY_CATEGORY["compute"])

    if occupancy_pct is not None and occupancy_pct < 50:
        patterns.extend(PATTERNS_BY_CATEGORY["occupancy"])

    # Always include general patterns
    patterns.extend([FUSE_ELEMENTWISE, PRECOMPUTE_INVARIANTS])

    # Deduplicate and sort by priority
    seen = set()
    unique = []
    for p in patterns:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)

    return sorted(unique, key=lambda p: p.priority)


def patterns_to_prompt(patterns: list[OptimizationPattern]) -> str:
    """Format patterns as context for the LLM prompt."""
    lines = ["## Applicable Optimization Patterns\n"]
    for i, p in enumerate(patterns, 1):
        lines.append(f"### Pattern {i}: {p.name} [{p.category}] (priority: {p.priority}/5)")
        lines.append(f"**When:** {p.condition}")
        lines.append(f"**How:** {p.transformation}")
        lines.append(f"**Impact:** {p.expected_impact}")
        if p.code_example:
            lines.append(f"**Example:**\n```python\n{p.code_example.strip()}\n```")
        lines.append("")

    return "\n".join(lines)
