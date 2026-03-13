"""
Enhanced tuner prompt: baseline's str_replace editing interface + KernForge enhancements.

The baseline tuner asks the LLM to make str_replace edits with zero guidance about
what's actually wrong. We inject NCU bottleneck diagnosis, optimization ladder phase,
and specific suggestions based on the kernel's performance profile.
"""

import re
from kernforge.eval.flashinfer_eval import EvalResult


_HARDWARE_INFO = """## Hardware Information

- GPU: NVIDIA {gpu_name}, {gpu_architecture} architecture.
"""

PROBLEM_STATEMENT = """## Problem Statement

You tune the custom Triton kernels to get better performance. Apply small, targeted edits
to the latest kernel version to fix correctness errors or improve performance.

"""

# ─── KernForge: bottleneck-aware guidance ───

BOTTLENECK_GUIDANCE = {
    "memory_bandwidth": """
## Performance Analysis: MEMORY BOUND

Your kernel is limited by memory bandwidth ({bandwidth_pct:.0f}% of peak).
Tensor cores are at {compute_pct:.0f}% utilization.

Priority edits:
1. **Reduce HBM traffic**: Can you fuse operations to avoid intermediate stores?
2. **Improve coalescing**: Ensure threads access consecutive addresses
3. **Increase reuse**: Can you load data once and use it multiple times?
4. **Vectorize loads**: Ensure BLOCK sizes are multiples of 4 for 128-bit loads
5. **Software pipelining**: Increase num_stages if SRAM budget allows
""",
    "compute": """
## Performance Analysis: COMPUTE BOUND

Your kernel is limited by compute throughput ({compute_pct:.0f}% of peak).
Memory bandwidth at {bandwidth_pct:.0f}%.

Priority edits:
1. **Use tl.dot()**: Replace any manual matmul (tl.sum(a * b)) with tl.dot()
2. **Tensor cores**: Ensure K dimensions are multiples of 16
3. **Reduce redundant math**: Move loop-invariant computations outside loops
4. **Precompute**: Compute shared values once (e.g., alpha*beta*k)
""",
    "low_occupancy": """
## Performance Analysis: LOW OCCUPANCY ({occupancy_pct:.0f}%)

Your kernel has low SM utilization — not enough warps to hide memory latency.

Priority edits:
1. **Reduce register pressure**: Smaller tile sizes, fewer live variables
2. **Increase num_warps**: More threads per block
3. **Reduce shared memory**: Free up space so more blocks can run per SM
4. **Registers per thread**: {regs_per_thread} — target < 128 for >25% occupancy
""",
    "register_spills": """
## Performance Analysis: REGISTER SPILLS DETECTED

Your kernel spills {spill_bytes} bytes to local memory — this is extremely slow.

Priority edits:
1. **REDUCE TILE SIZE IMMEDIATELY**: This is the most common fix
2. **Use fewer accumulators**: Recompute instead of storing
3. **Process tiles sequentially**: Instead of loading everything at once
4. **Simplify inner loop**: Move complex computations to a separate function
""",
}

CORRECTNESS_GUIDANCE = """
## Correctness Error Analysis

Your kernel produced incorrect results. Common causes:

1. **Missing masks**: Check every tl.load() and tl.store() — add mask= for variable dims
2. **bf16 accumulator**: Change tl.zeros(..., dtype=tl.bfloat16) → dtype=tl.float32
3. **Wrong stride**: Don't assume contiguous layout — pass strides as arguments
4. **Grid mismatch**: Verify grid dimensions match tl.program_id() usage
5. **tl.dot K-dimension**: Must be multiple of 16 for tensor cores
6. **Off-by-one in tiling**: Check boundary conditions for non-divisible dimensions
"""

COMPILE_GUIDANCE = """
## Compilation Error Analysis

Your kernel failed to compile. The error was:
```
{error}
```

Common fixes:
1. **torch.* inside @triton.jit**: Replace with tl.* (tl.exp, tl.sigmoid, etc.)
2. **numpy inside kernel**: Replace with tl.* operations
3. **Invalid syntax**: Check for hex floats (0x1.234p5), use decimal instead
4. **Missing import**: Ensure `import triton` and `import triton.language as tl`
5. **tl.dot with 1D args**: Reshape to 2D (e.g., x[:, None])
"""


TASK_INSTRUCTION = """## Task Instruction

Kernel definition:
```python
{definition}
```

The input shapes and dtype are: {dtype_str}

### Test Conditions
- **Correctness**: Verified against reference implementation
- **Warm-up**: 3 runs (autotuning happens here)
- **Performance**: 100 runs for timing

### Edit Format

Apply targeted str_replace edits. Keep the high-level architecture unchanged.

CRITICAL RULES:
1. `old_str` must match EXACTLY (including whitespace and indentation)
2. `old_str` must be unique in the file (include 3-5 lines of context)
3. `new_str` must be different from `old_str`
4. Send ALL edits in a single response

Output format:

<reasoning_1>
// why this edit improves the kernel
</reasoning_1>
<old_str_1>
// exact code to replace
</old_str_1>
<new_str_1>
// improved code
</new_str_1>

### Previous Kernels and Metrics:

<Previous Kernels and Metrics>
{previous_kernels_and_metrics}
</Previous Kernels and Metrics>
"""


def _is_correct_metric(metric) -> bool:
    """Check if a metric indicates correctness."""
    if isinstance(metric, EvalResult):
        return metric.correct
    elif isinstance(metric, str):
        return "correctness=True" in metric or '"correctness": true' in metric.lower()
    return False


def generate_tuner_prompt(
    previous_kernels: list[str],
    previous_metrics: list,
    task_params: dict,
    filter_wrong_attempts: bool = False,
    *,
    ncu_profile=None,
    ladder_context: str = None,
) -> str:
    """
    Generate the tuner prompt with KernForge enhancements.

    Compatible with baseline interface, plus:
    - NCU bottleneck diagnosis (if profile provided)
    - Optimization ladder phase context
    - Targeted guidance based on error type
    """
    # Filter wrong attempts if requested
    if filter_wrong_attempts and previous_kernels and previous_metrics:
        filtered = [
            (k, m) for k, m in zip(previous_kernels, previous_metrics)
            if _is_correct_metric(m)
        ]
        if filtered:
            previous_kernels, previous_metrics = zip(*filtered)
            previous_kernels = list(previous_kernels)
            previous_metrics = list(previous_metrics)
        else:
            previous_kernels, previous_metrics = [], []

    # Format previous kernels and metrics
    previous_str = "\n".join(
        f"\n### {i}-th attempt:\n\n```python\n{kernel}\n```\n\n"
        f"### {i}-th Runtime Metrics:\n{metric}"
        for i, (kernel, metric) in enumerate(zip(previous_kernels, previous_metrics))
    )

    # Build prompt
    prompt = PROBLEM_STATEMENT

    # Hardware info
    gpu_name = task_params.get("gpu_name")
    gpu_arch = task_params.get("gpu_architecture")
    if gpu_name and gpu_arch:
        prompt += _HARDWARE_INFO.format(gpu_name=gpu_name, gpu_architecture=gpu_arch)

    # KernForge: targeted guidance based on last eval result
    last_metric = previous_metrics[-1] if previous_metrics else None
    if last_metric:
        prompt += _generate_targeted_guidance(last_metric, ncu_profile)

    # KernForge: optimization ladder context
    if ladder_context:
        prompt += "\n" + ladder_context + "\n"

    # Task instruction with previous kernels
    format_dict = dict(task_params)
    format_dict["previous_kernels_and_metrics"] = previous_str

    prompt += TASK_INSTRUCTION.format(**format_dict)
    return prompt


def _generate_targeted_guidance(metric, ncu_profile=None) -> str:
    """Generate specific guidance based on the last eval result."""
    if isinstance(metric, EvalResult):
        if not metric.compiled:
            return COMPILE_GUIDANCE.format(error=metric.error or "Unknown error")
        if not metric.correct:
            return CORRECTNESS_GUIDANCE

        # Kernel is correct — provide performance guidance
        if ncu_profile is not None:
            return _format_ncu_guidance(ncu_profile)

        # No NCU data — generic performance hint
        if metric.speedup < 1.0:
            return """
## Performance Note
Your kernel is slower than the reference (speedup: {:.2f}x).
Focus on: using tl.dot() for matmuls, adding @triton.autotune, fusing operations.
""".format(metric.speedup)

    return ""


def _format_ncu_guidance(profile) -> str:
    """Format NCU profile into bottleneck-specific guidance."""
    bottleneck = getattr(profile, "bottleneck", None)
    if not bottleneck:
        return ""

    template = BOTTLENECK_GUIDANCE.get(bottleneck, "")
    if not template:
        return ""

    return template.format(
        bandwidth_pct=getattr(profile, "dram_throughput_pct", 0),
        compute_pct=getattr(profile, "sm_throughput_pct", 0),
        occupancy_pct=getattr(profile, "achieved_occupancy_pct", 0),
        regs_per_thread=getattr(profile, "registers_per_thread", 0),
        spill_bytes=getattr(profile, "local_load_bytes", 0) + getattr(profile, "local_store_bytes", 0),
    )
