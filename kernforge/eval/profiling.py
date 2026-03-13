"""
GPU profiling integration using flashinfer_bench.agents official tools.

Instead of our custom NCU runner, we use the official flashinfer_bench tools:
- flashinfer_bench_run_ncu: NCU profiling with Solution/Workload objects
- flashinfer_bench_run_sanitizer: Memory/race/sync/init checking

These are the SAME tools used by the contest organizers for evaluation.
"""

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


def create_solution_object(
    kernel_code: str,
    task_id: str,
    name: str = "kernforge_profile",
    language: str = "triton",
    entry_point: str = "main.py::run",
    destination_passing_style: bool = False,
):
    """Create a flashinfer_bench Solution object from kernel code."""
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

    lang = SupportedLanguages.TRITON if language == "triton" else SupportedLanguages.CUDA
    return Solution(
        name=name,
        definition=task_id,
        author="kernforge",
        spec=BuildSpec(
            language=lang,
            target_hardware=["cuda"],
            entry_point=entry_point,
            dependencies=[],
            destination_passing_style=destination_passing_style,
        ),
        sources=[SourceFile(path="main.py", content=kernel_code)],
    )


def get_workloads_for_task(dataset_root: str, task_id: str):
    """Get workload objects for a task from the dataset."""
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet.from_path(dataset_root)
    workloads = trace_set.workloads.get(task_id, [])
    return workloads


def run_ncu_profile(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
    *,
    ncu_set: str = "detailed",
    ncu_page: str = "details",
    max_lines: int = 200,
    timeout: int = 120,
) -> dict:
    """
    Run NCU profiling on a kernel using the official flashinfer_bench tool.

    Returns a dict with:
      - raw_output: full NCU text output
      - bottleneck: detected bottleneck type ('memory_bandwidth', 'compute', 'low_occupancy', etc.)
      - metrics: parsed key metrics
      - error: error message if profiling failed
    """
    try:
        from flashinfer_bench.agents import flashinfer_bench_run_ncu
    except ImportError:
        return {"error": "flashinfer_bench.agents not available"}

    solution = create_solution_object(kernel_code, task_id)
    workloads = get_workloads_for_task(dataset_root, task_id)

    if not workloads:
        return {"error": f"No workloads found for task {task_id}"}

    # Profile with the first workload (representative)
    workload = workloads[0]

    output = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        set=ncu_set,
        page=ncu_page,
        trace_set_path=dataset_root,
        timeout=timeout,
        max_lines=max_lines,
    )

    if output.startswith("ERROR:"):
        return {"error": output, "raw_output": output}

    # Parse the NCU output to extract key metrics
    metrics = _parse_ncu_output(output)
    bottleneck = _classify_bottleneck(metrics)

    return {
        "raw_output": output,
        "bottleneck": bottleneck,
        "metrics": metrics,
        "error": None,
    }


def run_sanitizer(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
    *,
    sanitizer_types: list[str] = None,
    timeout: int = 300,
    max_lines: int = 100,
) -> dict:
    """
    Run compute-sanitizer on a kernel using the official flashinfer_bench tool.

    Returns a dict with:
      - raw_output: full sanitizer text output
      - passed: True if all sanitizer checks passed
      - issues: list of detected issues
      - error: error message if sanitizer failed
    """
    try:
        from flashinfer_bench.agents import flashinfer_bench_run_sanitizer
    except ImportError:
        return {"error": "flashinfer_bench.agents not available", "passed": False}

    if sanitizer_types is None:
        sanitizer_types = ["memcheck", "racecheck"]  # Most important ones

    solution = create_solution_object(kernel_code, task_id)
    workloads = get_workloads_for_task(dataset_root, task_id)

    if not workloads:
        return {"error": f"No workloads found for task {task_id}", "passed": False}

    workload = workloads[0]

    output = flashinfer_bench_run_sanitizer(
        solution=solution,
        workload=workload,
        sanitizer_types=sanitizer_types,
        trace_set_path=dataset_root,
        timeout=timeout,
        max_lines=max_lines,
    )

    if output.startswith("ERROR:"):
        return {"error": output, "raw_output": output, "passed": False}

    # Parse sanitizer output
    passed = "detected issues" not in output.lower()
    issues = _parse_sanitizer_issues(output)

    return {
        "raw_output": output,
        "passed": passed,
        "issues": issues,
        "error": None,
    }


def format_ncu_for_prompt(ncu_result: dict) -> str:
    """Format NCU profiling results for inclusion in LLM prompt."""
    if ncu_result.get("error"):
        return f"[NCU profiling failed: {ncu_result['error']}]"

    bottleneck = ncu_result.get("bottleneck", "unknown")
    metrics = ncu_result.get("metrics", {})

    lines = ["\n## NCU Performance Profile\n"]

    if bottleneck == "memory_bandwidth":
        lines.append(f"**Bottleneck: MEMORY BOUND** ({metrics.get('dram_pct', '?')}% DRAM bandwidth)")
        lines.append("Priority: reduce HBM traffic, improve coalescing, fuse operations")
    elif bottleneck == "compute":
        lines.append(f"**Bottleneck: COMPUTE BOUND** ({metrics.get('sm_pct', '?')}% SM throughput)")
        lines.append("Priority: use tl.dot() for matmuls, reduce redundant math")
    elif bottleneck == "low_occupancy":
        lines.append(f"**Bottleneck: LOW OCCUPANCY** ({metrics.get('occupancy_pct', '?')}%)")
        lines.append("Priority: reduce register pressure, decrease tile sizes")
    elif bottleneck == "register_spills":
        lines.append("**Bottleneck: REGISTER SPILLS** — memory spills to local memory")
        lines.append("Priority: REDUCE TILE SIZE immediately, simplify inner loop")
    else:
        lines.append("**No clear bottleneck detected** — kernel is reasonably balanced")

    # Key metrics
    if metrics:
        lines.append("\nKey metrics:")
        for key, val in metrics.items():
            lines.append(f"  {key}: {val}")

    return "\n".join(lines)


def format_sanitizer_for_prompt(sanitizer_result: dict) -> str:
    """Format sanitizer results for inclusion in LLM prompt."""
    if sanitizer_result.get("error"):
        return ""  # Don't add noise if sanitizer failed

    if sanitizer_result.get("passed"):
        return ""  # No issues, don't mention it

    issues = sanitizer_result.get("issues", [])
    if not issues:
        return ""

    lines = ["\n## Memory Safety Issues Detected\n"]
    for issue in issues[:5]:
        lines.append(f"  ⚠️ {issue}")
    lines.append("\nFix these memory safety issues before optimizing for performance.")
    return "\n".join(lines)


# ─── NCU output parsing ───

def _parse_ncu_output(output: str) -> dict:
    """Parse NCU text output to extract key metrics."""
    metrics = {}

    patterns = {
        "dram_pct": r"DRAM\s+Throughput\s+(\d+\.?\d*)%",
        "sm_pct": r"SM\s+(?:Active\s+)?Throughput\s+(\d+\.?\d*)%",
        "occupancy_pct": r"Achieved\s+Occupancy\s+(\d+\.?\d*)%",
        "regs_per_thread": r"Registers\s+Per\s+Thread\s+(\d+)",
        "shared_mem_bytes": r"Shared\s+Memory\s+(?:Configuration\s+)?Size\s+(\d+)",
        "local_load_bytes": r"Local\s+Load\s+Throughput.*?(\d+\.?\d*)\s*(?:byte|GB)",
        "local_store_bytes": r"Local\s+Store\s+Throughput.*?(\d+\.?\d*)\s*(?:byte|GB)",
        "l2_hit_rate": r"L2\s+(?:Cache\s+)?Hit\s+Rate\s+(\d+\.?\d*)%",
        "duration_us": r"Duration\s+(\d+\.?\d*)\s*(?:us|μs)",
        "grid_size": r"Grid\s+Size\s+\[(\d+),\s*(\d+),\s*(\d+)\]",
        "block_size": r"Block\s+Size\s+\[(\d+),\s*(\d+),\s*(\d+)\]",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            if key in ("grid_size", "block_size"):
                metrics[key] = f"[{match.group(1)}, {match.group(2)}, {match.group(3)}]"
            else:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    metrics[key] = match.group(1)

    return metrics


def _classify_bottleneck(metrics: dict) -> str:
    """Classify the performance bottleneck from parsed NCU metrics."""
    dram = metrics.get("dram_pct", 0)
    sm = metrics.get("sm_pct", 0)
    occupancy = metrics.get("occupancy_pct", 0)
    local_load = metrics.get("local_load_bytes", 0)
    local_store = metrics.get("local_store_bytes", 0)

    # Register spills are the worst
    if local_load > 0 or local_store > 0:
        return "register_spills"

    # Low occupancy
    if occupancy > 0 and occupancy < 25:
        return "low_occupancy"

    # Memory vs compute bound
    if dram > 0 and sm > 0:
        if dram > sm * 1.5:
            return "memory_bandwidth"
        if sm > dram * 1.5:
            return "compute"

    # Single metric dominance
    if dram > 60:
        return "memory_bandwidth"
    if sm > 60:
        return "compute"
    if occupancy > 0 and occupancy < 40:
        return "low_occupancy"

    return "balanced"


def _parse_sanitizer_issues(output: str) -> list[str]:
    """Parse sanitizer output to extract issue descriptions."""
    issues = []
    for line in output.split("\n"):
        line = line.strip()
        if any(keyword in line.lower() for keyword in
               ["error", "warning", "invalid", "race", "uninitialized", "out of range"]):
            if not line.startswith("=") and len(line) > 10:
                issues.append(line)
    return issues[:10]  # Limit to 10 issues
