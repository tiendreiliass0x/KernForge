"""
Optimization Ladder — structured progression from naive to competitive.

Inspired by Wafer's 7-step topk_sigmoid optimization:
  naive → parallel → shared mem → warp shuffle → DPP → DPP broadcast → hand-tuned ASM

Instead of letting the agent randomly explore, we define a LADDER of increasing
sophistication. Each rung focuses on ONE class of optimization. The agent climbs
the ladder, and at each rung it generates the best kernel it can using that
technique before moving up.

This is fundamentally better than random mutation because:
1. It ensures the agent tries the high-impact optimizations first
2. It prevents the agent from getting stuck on micro-optimizations before
   the algorithmic structure is right
3. Each rung's output becomes the starting point for the next rung
4. The agent has a clear goal at each step ("use tensor cores" not "make it faster")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LadderRung:
    """One step in the optimization ladder."""
    name: str
    phase: int  # 1=correctness, 2=algorithmic, 3=hardware, 4=micro
    goal: str
    instruction: str
    success_criteria: str
    max_attempts: int = 5  # max iterations at this rung before moving on
    skip_if: str = ""  # condition to skip this rung


# =============================================================================
# GDN DECODE optimization ladder
# =============================================================================

GDN_DECODE_LADDER = [
    LadderRung(
        name="naive_correct",
        phase=1,
        goal="Get a correct kernel, no optimization",
        instruction="""Write a straightforward Triton kernel that exactly matches the reference.
One thread block per (batch, v_head). Each block loads the entire 128x128 state,
computes the full recurrence, and writes back.
Do NOT optimize. Priority is correctness.
Use float32 for ALL arithmetic. Cast output to bf16 only at the final store.""",
        success_criteria="Passes correctness check on all test shapes",
    ),

    LadderRung(
        name="tiled_state",
        phase=2,
        goal="Tile the state matrix to fit in registers/SRAM",
        instruction="""The 128x128 f32 state is 64KB — it fits in B200 SRAM but not in registers.
Tile the state along the V dimension (rows) with BLOCK_V=32 or 64.
Process K=128 columns in full (contiguous in k-last layout).

Each tile: [BLOCK_V, 128] f32 = 16-32KB
Operations per tile:
  1. Load state tile
  2. Compute Sk_partial = tile @ k  (partial matvec)
  3. Apply decay: tile *= alpha
  4. Subtract: tile -= alpha*beta * Sk_partial[:,None] * k[None,:]
  5. Add: tile += dt * v_tile[:,None] * k[None,:]
  6. Compute output_partial = tile @ q  (partial matvec)
  7. Store state tile

Fuse ALL 6 operations in one pass per tile (no intermediate HBM writes).""",
        success_criteria="Correct AND state operations fused in single tiled loop",
    ),

    LadderRung(
        name="tensor_cores",
        phase=2,
        goal="Use tl.dot() for matrix operations to leverage tensor cores",
        instruction="""Replace scalar matrix-vector products with tl.dot() where possible.

Key operations to convert:
  - S_tile @ k: reshape k to [128, 1], use tl.dot(s_tile, k_2d) → [BLOCK_V, 1]
  - S_tile @ q: same pattern
  - v_tile @ k^T: reshape v to [BLOCK_V, 1] and k to [1, 128], use tl.dot

Requirements for tl.dot:
  - Both inputs must be 2D
  - Inner dimension must be multiple of 16
  - Inputs should be bf16 for maximum throughput (but accumulate in f32)

For the state operations, since state is f32, you may need to cast tiles to bf16
for the tl.dot and accumulate back in f32. Alternatively, keep scalar ops for
the state update but use tl.dot for the output computation.""",
        success_criteria="At least the output computation (S @ q) uses tl.dot",
        skip_if="State arithmetic is f32-only and doesn't benefit from tensor cores",
    ),

    LadderRung(
        name="autotune",
        phase=3,
        goal="Add @triton.autotune with a good configuration space",
        instruction="""Add @triton.autotune with 5-8 configurations exploring:
  - BLOCK_V: [16, 32, 64, 128] (rows of state per block)
  - num_warps: [4, 8] (parallelism within block)
  - num_stages: [1, 2, 3] (software pipelining)

Key: use `key=['batch_size']` so the best config is selected per batch size.
Small batches prefer larger BLOCK_V (more work per block), large batches
prefer smaller BLOCK_V (more blocks for SM utilization).

Include at least one config with BLOCK_V=128 (processes entire row in one block)
and one with BLOCK_V=32 (maximizes SM utilization for large batches).""",
        success_criteria="Autotune configs present, kernel auto-selects best for workload",
    ),

    LadderRung(
        name="gva_optimization",
        phase=3,
        goal="Optimize GVA head mapping to avoid redundant q/k loads",
        instruction="""With GVA ratio=2, v_heads 2i and 2i+1 share the same q/k head i.

Current approach likely uses repeat_interleave or loads q/k separately per v_head.
Better: process pairs of v_heads together, loading q/k once.

Option A: Grid over q/k heads, inner loop over the 2 corresponding v_heads
Option B: Grid over v_heads, but cache q/k in shared memory with a "load once" pattern

The q and k vectors are only 128*2=256 bytes each (bf16) — trivially small.
The optimization here is reducing redundant computation, not memory savings.""",
        success_criteria="q/k loaded once per GVA group, not once per v_head",
    ),

    LadderRung(
        name="memory_access_pattern",
        phase=3,
        goal="Ensure coalesced memory access for state read/write",
        instruction="""State layout is [B, H, V=128, K=128] with K contiguous.

Verify that:
1. State loads read contiguous K elements (not strided)
2. State stores write contiguous K elements
3. Each warp's threads access consecutive memory addresses

In Triton, this means the innermost tl.arange should index the K dimension.
The V dimension should be indexed by the outer arange or program_id.

If tiling [BLOCK_V, 128]: each thread block reads BLOCK_V contiguous 128-element
rows. Ensure the tl.load pointer arithmetic gives contiguous access within each row.

Also: use tl.load with eviction_policy='evict_last' for state since it will be
written back, and 'evict_first' for inputs that are used once.""",
        success_criteria="Memory access pattern confirmed coalesced via ISA analysis",
    ),

    LadderRung(
        name="precompute_and_fuse",
        phase=4,
        goal="Precompute all gate values and fuse remaining redundant operations",
        instruction="""Micro-optimization pass:
1. Precompute alpha = sigmoid(A_log[h]) ONCE per head (it's the same for all batches in the same head — but A_log is per-head, not per-batch)
2. Precompute k_scaled = alpha * beta * k and dt_k = dt * k to reduce per-tile multiplications
3. In the tile loop, the update becomes:
   Sk = tile @ k
   tile = alpha * tile - Sk[:,None] * k_scaled[None,:] + v[:,None] * dt_k[None,:]
   This saves 2 multiplications per element per tile.

4. If doing tl.dot for S@q, pipeline the load of the next state tile while computing
   the current tile's output (software pipelining via num_stages).

5. Consider processing the state in a strided pattern to avoid bank conflicts
   in shared memory (if using shared memory explicitly).""",
        success_criteria="Gate values precomputed, per-tile ops minimized",
    ),

    LadderRung(
        name="inline_ptx",
        phase=4,
        goal="Use inline PTX for fast transcendentals and packed operations",
        instruction="""Triton's `tl.inline_asm_elementwise()` lets you inject PTX instructions
for operations where Triton's compiler doesn't generate optimal code.

Candidates for GDN decode:
1. **Fast sigmoid**: sigmoid(x) = rcp.approx(1 + ex2.approx(-x))
   - Used for alpha (sigmoid(A_log)) and beta (sigmoid(b))
   - The approx variants are ~2x faster than the full precision path
   - Only use if correctness check passes (approx loses ~1 ULP)

2. **Fast softplus**: softplus(x) = x + log1p(exp(-|x|))
   - For the dt computation: softplus(a + dt_bias)
   - Use lg2.approx + ex2.approx for fast log/exp

3. **Packed bf16x2**: When processing GVA head pairs (ratio=2),
   pack two bf16 values and use fma.rn.bf16x2 for paired operations.

API:
```python
(y,) = tl.inline_asm_elementwise(
    asm="rcp.approx.ftz.f32 $0, $1;",
    constraints="=r,r",
    args=[x],
    dtype=[tl.float32],
    is_pure=True,
    pack=1,
)
```

IMPORTANT: Always verify correctness after adding inline PTX. Approximate
instructions can break numerical tolerances. If correctness fails, revert.""",
        success_criteria="Inline PTX used for at least one transcendental, correctness maintained",
        skip_if="Kernel already meets target latency without PTX tricks",
    ),

    LadderRung(
        name="b200_advanced",
        phase=4,
        goal="Apply B200-specific advanced patterns: persistent kernel, TMA pipelining, warp specialization",
        instruction="""Apply the most advanced B200 patterns:

1. **Persistent kernel**: Instead of one kernel launch per (batch, head),
   launch one persistent kernel where each SM loops over its assigned
   (batch, head) pairs. State stays in SRAM between pairs.
   Grid = (num_SMs,) with work-stealing or static assignment.

2. **TMA pipelining**: Use num_stages=3-4 to overlap state tile loads
   with computation on previously loaded tiles. B200's 256KB SRAM fits:
   - 3 × 64KB state tiles (for 3 pipeline stages)
   - Plus input vectors and accumulators (~4KB)
   - Total: ~196KB out of 256KB available

3. **Warp specialization** (advanced, may need CUDA):
   - Warp 0: TMA producer (loads next state tile)
   - Warps 1-7: MMA consumers (compute on current tile)
   - Synchronized via pipeline barriers
   
   In Triton, approximate this by:
   - Using num_stages > 1 for software pipelining
   - Processing the state in a streaming fashion

4. **For variable-length batches (prefill)**:
   Use the grouped scheduling pattern — precompute a tile→sequence mapping,
   only update pointers when the sequence changes.

The persistent kernel pattern is the highest-impact change at this stage.
It eliminates kernel launch overhead and keeps state in fast memory.""",
        success_criteria="Persistent kernel structure or multi-stage pipelining active",
    ),
]


# =============================================================================
# GDN PREFILL optimization ladder
# =============================================================================

GDN_PREFILL_LADDER = [
    LadderRung(
        name="naive_sequential",
        phase=1,
        goal="Correct sequential prefill — process tokens one at a time",
        instruction="""Start with a simple loop over the sequence.
For each token, apply the decode recurrence.
This is correct but O(T * D²) and fully sequential.
Use this as the correctness baseline.""",
        success_criteria="Matches reference output for variable-length batches",
    ),

    LadderRung(
        name="chunkwise_parallel",
        phase=2,
        goal="Implement chunkwise parallel form",
        instruction="""Split each sequence into chunks of C=64 tokens.
Within each chunk, use the WY representation to parallelize Householder products.
Between chunks, sequential state scan (but there are only L/64 chunks).

Key: adapt from FLA's working implementation in
fla-org/flash-linear-attention/fla/ops/gated_delta_rule/

The three-phase approach:
1. Intra-chunk: WY factors + local attention + partial state updates
2. Inter-chunk: sequential state scan S_c = decay * S_{c-1} * P_c + U_c
3. Output: combine intra-chunk and cross-chunk contributions""",
        success_criteria="Correct with chunkwise parallelism, speedup over sequential",
    ),

    LadderRung(
        name="fused_intra_chunk",
        phase=3,
        goal="Fuse intra-chunk computation into a single kernel",
        instruction="""The intra-chunk phase has three sub-computations that can be fused:
1. WY factor computation
2. Chunk-local attention
3. Partial state update

Fuse these into one kernel that processes each chunk without intermediate HBM writes.
Each block handles one (batch, head, chunk) triple.""",
        success_criteria="Intra-chunk computation fused, no intermediate HBM traffic",
    ),

    LadderRung(
        name="variable_length_handling",
        phase=3,
        goal="Efficient variable-length sequence handling via cu_seqlens",
        instruction="""cu_seqlens defines the packed sequence boundaries.
Key: binary search or precomputed mapping from chunk_id to sequence boundaries.
Handle edge cases: chunks that cross sequence boundaries must NOT leak state.""",
        success_criteria="Correct for variable-length batches with different sequence lengths",
    ),
]


# =============================================================================
# Generic ladder for unknown kernel types
# =============================================================================

GENERIC_LADDER = [
    LadderRung(name="naive_correct", phase=1,
               goal="Correct baseline",
               instruction="Write a correct kernel matching the reference exactly.",
               success_criteria="Passes correctness checks"),
    LadderRung(name="tiled", phase=2,
               goal="Tile the computation to fit in SRAM",
               instruction="Identify the largest data structure and tile it to fit in B200's 256KB SRAM.",
               success_criteria="Data reuse via tiling"),
    LadderRung(name="tensor_cores", phase=2,
               goal="Use tensor cores for matrix operations",
               instruction="Replace element-wise matrix ops with tl.dot().",
               success_criteria="HMMA instructions in ISA"),
    LadderRung(name="autotune", phase=3,
               goal="Autotune tile sizes and launch config",
               instruction="Add @triton.autotune with 5+ configurations.",
               success_criteria="Autotune present"),
    LadderRung(name="fuse_and_optimize", phase=4,
               goal="Fuse operations and apply micro-optimizations",
               instruction="Precompute invariants, fuse element-wise ops, optimize memory access.",
               success_criteria="Reduced instruction count"),
]


# =============================================================================
# Ladder selection and management
# =============================================================================

LADDERS = {
    "gdn_decode": GDN_DECODE_LADDER,
    "gdn_prefill": GDN_PREFILL_LADDER,
    "gdn": GDN_DECODE_LADDER,  # default to decode
    "generic": GENERIC_LADDER,
}


def get_ladder(kernel_name: str) -> list[LadderRung]:
    """Select the appropriate optimization ladder for a kernel."""
    name_lower = kernel_name.lower()
    if "decode" in name_lower and ("gdn" in name_lower or "gated_delta" in name_lower):
        return GDN_DECODE_LADDER
    elif "prefill" in name_lower and ("gdn" in name_lower or "gated_delta" in name_lower):
        return GDN_PREFILL_LADDER
    elif "gdn" in name_lower or "gated_delta" in name_lower:
        return GDN_DECODE_LADDER
    return GENERIC_LADDER


def get_rung_for_generation(
    kernel_name: str,
    generation: int,
    is_correct: bool,
    current_rung_idx: int = 0,
) -> tuple[LadderRung, int]:
    """
    Determine which rung of the ladder to use for this generation.

    Returns (rung, rung_index) so the caller can track progress up the ladder.
    """
    ladder = get_ladder(kernel_name)

    # If not correct, stay on current rung (or go back to rung 0)
    if not is_correct:
        rung_idx = min(current_rung_idx, len(ladder) - 1)
        return ladder[rung_idx], rung_idx

    # If correct, advance to next rung
    next_idx = min(current_rung_idx + 1, len(ladder) - 1)
    return ladder[next_idx], next_idx


def rung_to_prompt(rung: LadderRung, rung_idx: int, total_rungs: int) -> str:
    """Format a ladder rung as context for the LLM."""
    return f"""## Optimization Ladder — Step {rung_idx + 1}/{total_rungs}: {rung.name}

**Phase {rung.phase}** — {"Correctness" if rung.phase == 1 else "Algorithmic" if rung.phase == 2 else "Hardware" if rung.phase == 3 else "Micro-optimization"}

**Goal:** {rung.goal}

**Instructions:**
{rung.instruction}

**Success Criteria:** {rung.success_criteria}

IMPORTANT: Focus ONLY on this step's goal. Do not jump ahead to later optimizations.
Get this step right first, then we'll advance.
"""
