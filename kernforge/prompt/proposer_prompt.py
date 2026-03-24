"""
Enhanced proposer prompt: baseline structure + KernForge domain knowledge.

Compatible with the official baseline's generate_proposer_prompt() interface,
but injects optimization knowledge, common bug warnings, and kernel-type-specific
guidance that the baseline completely lacks.
"""

from typing import Optional

# ─── Baseline prompt components (kept for compatibility) ───

PROBLEM_STATEMENT = """## Problem Statement

You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace.
You may replace multiple operators with custom implementations,
consider operator fusion opportunities (combining multiple operators into a single kernel),
or algorithmic changes (such as online softmax). You are only limited by your imagination.

The goal is to get the best performance for the given architecture.

"""

POOL_PROMPT = """## Kernel Pool

Here are previous kernel attempts and their runtime metrics. Use them as reference.

DO NOT directly copy the kernels. Generate your own with major modifications:
fix correctness errors, change algorithms, improve performance.

<Kernels and their Runtime Metrics>
{pool_kernels_and_metrics}
</Kernels and their Runtime Metrics>

Now generate your own kernel, improving on the above:
"""

TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

"""

# ─── KernForge enhancement: domain knowledge injection ───

TRITON_REQUIREMENTS = """Requirements:
- Write clean, efficient Triton code optimized for {target_gpu} ({gpu_architecture})
- Use modern Triton syntax with proper grid computation
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized
- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations

IMPORTANT SYNTAX RULES:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- All code must be valid Python that passes ast.parse()
- NO C/CUDA specific syntax - this is Python/Triton code
"""

COMMON_BUGS = """
## Critical Triton Bugs to Avoid

1. **Missing masks**: ALWAYS use `mask=` on `tl.load()` and `tl.store()` for variable-size dimensions
   ```python
   x = tl.load(ptr + offs, mask=offs < N, other=0.0)  # CORRECT
   x = tl.load(ptr + offs)  # WRONG — crashes on boundary tiles
   ```

2. **bf16 accumulators**: ALWAYS accumulate in float32, cast only at final store
   ```python
   acc = tl.zeros([M, N], dtype=tl.float32)  # CORRECT
   acc = tl.zeros([M, N], dtype=tl.bfloat16)  # WRONG — catastrophic drift
   ```

3. **tl.dot dimensions**: Requires 2D inputs with K dimension multiple of 16
   ```python
   c = tl.dot(a, b)  # a must be [M,K], b must be [K,N], K % 16 == 0
   ```

4. **No torch/numpy in @triton.jit**: Use tl.* equivalents (tl.exp, tl.sigmoid, tl.sqrt)

5. **Always pass strides as kernel arguments**: Never assume contiguous memory layout

6. **Grid must match tl.program_id usage**: If kernel uses program_id(0) and program_id(1), grid must be 2D
"""

GPU_KNOWLEDGE = {
    "B200": """
## B200 (Blackwell) Architecture Notes
- HBM: 192GB, 8 TB/s bandwidth
- SRAM: 256 KB per SM (use for staging data)
- BF16 tensor core peak: 2,250 TFLOPS
- 192 SMs — launch enough blocks to fill them (≥192 for full occupancy)
- Registers: 65,536 per SM — keep per-thread usage < 128 for >25% occupancy
- TMA available for async bulk memory copies (Triton handles via num_stages)
- Arithmetic intensity crossover: ~280 FLOPs/byte
""",
    "H100": """
## H100 (Hopper) Architecture Notes
- HBM: 80GB, 3.35 TB/s bandwidth
- SRAM: 228 KB per SM
- BF16 tensor core peak: 990 TFLOPS
- 132 SMs
- TMA available (Triton handles via num_stages)
""",
    "A100": """
## A100 (Ampere) Architecture Notes
- HBM: 80GB, 2 TB/s bandwidth
- SRAM: 192 KB per SM
- BF16 tensor core peak: 312 TFLOPS
- 108 SMs
""",
}

OPTIMIZATION_GUIDANCE = """
## Optimization Strategy

Follow this progression (don't skip steps):

1. **Correctness first**: Get a working kernel before optimizing
2. **Use tl.dot()** for ALL matrix multiplications (10-50x faster than manual)
3. **Add @triton.autotune** with at least 4-5 configurations varying BLOCK sizes and num_warps
4. **Minimize HBM traffic**: Fuse operations, avoid intermediate writes
5. **Coalesced memory access**: Fastest-varying dimension should be contiguous across threads

### Autotune Template
```python
@triton.autotune(
    configs=[
        triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 64}}, num_warps=4, num_stages=2),
        triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 64}}, num_warps=4, num_stages=3),
        triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 128}}, num_warps=8, num_stages=2),
        triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 128}}, num_warps=8, num_stages=2),
    ],
    key=[...],  # dimensions that change at runtime
)
```
"""

# ─── Kernel-type-specific knowledge ───

KERNEL_TYPE_GUIDANCE = {
    "gdn": """
## GDN (Gated Delta Net) Specific Guidance

The GDN decode step has a recurrent state update:
  S_t = α · S_{t-1} · (I - β · k · k^T) + dt · v · k^T

Key patterns:
- **Grid**: (batch, v_head, v_tiles) — one block per state tile
- **GVA head mapping**: `qk_head = v_head // (H_v // H_q)` — integer division, NOT multiply
- **State is always float32** — never bf16 for state arithmetic
- **Precompute**: `ak = alpha * beta * k` and `dk = dt * k` outside the tile loop
- **Fused update**: Decay + erase + delta write in ONE pass over the state tile
- **K-last layout**: State stored as [batch, v_head, V, K] for coalesced access
""",
    "moe": """
## MOE (Mixture of Experts) Specific Guidance

Key patterns:
- **Routing**: Top-k expert selection, often with softmax + scatter
- **Block-scale FP8**: Quantized expert weights with per-block scaling
- **Fusion opportunity**: Fuse routing + expert compute + combine
- **Load balancing**: Experts may have unequal token counts — handle variable sizes
- **Memory**: Expert weights are large — minimize redundant loads
""",
    "gemm": """
## GEMM Specific Guidance

Key patterns:
- **Tile sizes**: Start with 128x128x32, tune from there
- **Tensor cores**: MUST use tl.dot() — this IS the computation
- **Software pipelining**: num_stages=2-4 for overlapping loads with compute
- **Split-K**: For tall-skinny matrices, split the K dimension across blocks
- **Epilogue fusion**: Fuse activation (ReLU, GELU) into the GEMM epilogue
""",
    "dsa_paged": """
## DSA (Paged Attention) Specific Guidance

Key patterns:
- **Page table**: Indirect access through page table mapping
- **Block size**: Usually 16 or 32 tokens per page
- **Softmax**: Online softmax (running max + running sum) for numerical stability
- **KV cache**: Paged layout — non-contiguous blocks, need page table lookup
- **Sequence parallelism**: Tile over sequence length, reduce across tiles
""",
}


# ─── Strategy context (from cross-run learning) ───


def format_strategy_context(strategy_db, kernel_type: str) -> str:
    """Format strategy DB into prompt context."""
    if strategy_db is None:
        return ""

    best = strategy_db.get_best_strategies(kernel_type, n=3)
    failed = strategy_db.get_failed_strategies(kernel_type, n=3)

    if not best and not failed:
        return ""

    parts = ["\n## Optimization History (from previous runs)\n"]
    if best:
        parts.append("Strategies that WORKED well (try these):")
        for s in best:
            strategy = getattr(s, "strategy", None)
            if strategy is None and isinstance(s, dict):
                strategy = s.get("strategy", "unknown")
            improvement = getattr(s, "improvement_pct", None)
            if improvement is None and isinstance(s, dict):
                improvement = s.get("improvement_pct", 0.0)
            parts.append(f"  - {strategy}: {float(improvement or 0.0):.1f}% faster")
    if failed:
        parts.append("\nStrategies that FAILED (avoid these):")
        for s in failed:
            strategy = getattr(s, "strategy", None)
            if strategy is None and isinstance(s, dict):
                strategy = s.get("strategy", "unknown")
            error_type = getattr(s, "error_type", None)
            if error_type is None and isinstance(s, dict):
                error_type = s.get("error_type", "unknown failure")
            parts.append(f"  - {strategy}: {error_type or 'unknown failure'}")

    return "\n".join(parts)


# ─── Main prompt generation ───


def generate_proposer_prompt(
    task_params: dict,
    pool_prompt: Optional[str] = None,
    *,
    kernel_type: Optional[str] = None,
    gpu_name: Optional[str] = None,
    strategy_db=None,
    corpus_context: Optional[str] = None,
    ladder_context: Optional[str] = None,
) -> str:
    """
    Generate the proposer prompt with KernForge enhancements.

    Compatible with baseline interface (task_params + pool_prompt),
    plus optional KernForge enhancements (domain knowledge, strategy DB, corpus).
    """
    gpu_name = gpu_name or task_params.get("gpu_name", "B200")
    gpu_arch = task_params.get("gpu_architecture", "Blackwell")

    prompt = PROBLEM_STATEMENT

    # Core task
    prompt += TRITON_PROMPT.format(
        target_gpu=gpu_name,
        definition=task_params["definition"],
    )

    # KernForge: requirements with bug warnings
    prompt += TRITON_REQUIREMENTS.format(
        target_gpu=gpu_name,
        gpu_architecture=gpu_arch,
    )
    prompt += COMMON_BUGS

    # KernForge: GPU architecture knowledge
    if gpu_name in GPU_KNOWLEDGE:
        prompt += GPU_KNOWLEDGE[gpu_name]

    # KernForge: kernel-type-specific guidance
    if kernel_type and kernel_type in KERNEL_TYPE_GUIDANCE:
        prompt += KERNEL_TYPE_GUIDANCE[kernel_type]

    # KernForge: optimization guidance
    prompt += OPTIMIZATION_GUIDANCE

    # KernForge: reference corpus (few-shot examples)
    if corpus_context:
        prompt += "\n## Reference Implementations\n\n" + corpus_context

    # KernForge: optimization ladder context
    if ladder_context:
        prompt += "\n" + ladder_context

    # KernForge: strategy history from previous runs
    if strategy_db:
        strategy_ctx = format_strategy_context(strategy_db, kernel_type or "general")
        if strategy_ctx:
            prompt += strategy_ctx

    # Pool of previous kernels (baseline compatible)
    if pool_prompt:
        prompt += "\n" + pool_prompt

    prompt += "\nGenerate the complete, runnable implementation:\n"
    return prompt


def generate_pool_prompt(
    *,
    kernel_pool: list,
    metrics_pool: list,
    kernel_pool_ids: list[int] | None = None,
    elite_kernel_pool: list | None = None,
    elite_metrics_pool: list | None = None,
    elite_pool_ids: list[int] | None = None,
) -> str:
    """Build pool prompt. Matches baseline interface exactly."""
    elite_kernel_pool = elite_kernel_pool or []
    elite_metrics_pool = elite_metrics_pool or []

    parts = []

    elite_part = _format_pool_single(
        elite_kernel_pool, elite_metrics_pool, elite_pool_ids
    )
    if elite_part:
        parts.append("## Context type: elite\n\n" + elite_part)

    recent_part = _format_pool_single(kernel_pool, metrics_pool, kernel_pool_ids)
    if recent_part:
        parts.append("## Context type: recent\n\n" + recent_part)

    if not parts:
        return ""

    merged = "\n\n".join(parts)
    return POOL_PROMPT.format(pool_kernels_and_metrics=merged)


def _format_pool_single(kernels, metrics, ids=None):
    if not kernels:
        return ""
    return "\n\n".join(
        f"### {i}-th kernel"
        + (f" (proposal_id={ids[i]})" if ids and i < len(ids) else "")
        + f":\n\n```python\n{kernel}\n```\n\n### {i}-th metrics:\n{metric}"
        for i, (kernel, metric) in enumerate(zip(kernels, metrics))
    )
