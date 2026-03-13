"""
FlashInfer-Bench agents integration.

The starter kit provides powerful APIs that the baseline completely ignores:
  - flashinfer_bench.agents.flashinfer_bench_run_ncu → NCU profiling
  - flashinfer_bench.agents.flashinfer_bench_run_sanitizer → memory/race checks
  - flashinfer_bench.agents.pack_solution_from_files → submission packaging

We use these to:
1. Get REAL NCU bottleneck data (not just "speedup: 0.7")
2. Catch memory errors that correctness tests miss
3. Package submissions correctly
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NCUProfile:
    """Parsed NCU profiling results."""
    bottleneck: str  # "memory_bandwidth", "compute", "low_occupancy", "register_spills", "unknown"
    dram_throughput_pct: float = 0.0
    sm_throughput_pct: float = 0.0
    achieved_occupancy_pct: float = 0.0
    registers_per_thread: int = 0
    local_load_bytes: int = 0
    local_store_bytes: int = 0
    raw_output: str = ""


def profile_kernel_ncu(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
) -> Optional[NCUProfile]:
    """
    Profile a kernel using flashinfer-bench's NCU integration.

    Returns parsed NCU profile or None if profiling fails.
    Only call this on CORRECT kernels that we want to optimize further.
    """
    try:
        from flashinfer_bench.data import (
            BuildSpec, Solution, SourceFile, SupportedLanguages, TraceSet,
        )
        from flashinfer_bench.agents import flashinfer_bench_run_ncu
    except ImportError:
        logger.debug("flashinfer_bench.agents not available, skipping NCU profiling")
        return None

    try:
        # Build solution object
        trace_set = TraceSet.from_path(dataset_root)
        solution = Solution(
            name=f"ncu_profile",
            definition=task_id,
            author="kernforge",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["cuda"],
                entry_point="main.py::run",
                dependencies=[],
                destination_passing_style=False,
            ),
            sources=[SourceFile(path="main.py", content=kernel_code)],
        )

        # Get first workload for this definition
        workloads = trace_set.workloads.get(task_id, [])
        if not workloads:
            logger.warning(f"No workloads found for {task_id}, skipping NCU")
            return None

        workload = workloads[0]

        # Run NCU via flashinfer-bench agents API
        ncu_output = flashinfer_bench_run_ncu(
            solution=solution,
            workload=workload,
            set="detailed",
            page="details",
            timeout=120,
        )

        return _parse_ncu_output(ncu_output)

    except Exception as e:
        logger.warning(f"NCU profiling failed: {e}")
        return None


def run_sanitizer(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
) -> Optional[str]:
    """
    Run CUDA sanitizers (memcheck, racecheck) on a kernel.

    Returns error description if issues found, None if clean.
    Call this on kernels that pass correctness but we suspect have subtle bugs.
    """
    try:
        from flashinfer_bench.data import (
            BuildSpec, Solution, SourceFile, SupportedLanguages, TraceSet,
        )
        from flashinfer_bench.agents import flashinfer_bench_run_sanitizer
    except ImportError:
        logger.debug("flashinfer_bench.agents not available, skipping sanitizer")
        return None

    try:
        trace_set = TraceSet.from_path(dataset_root)
        solution = Solution(
            name=f"sanitizer_check",
            definition=task_id,
            author="kernforge",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["cuda"],
                entry_point="main.py::run",
                dependencies=[],
                destination_passing_style=False,
            ),
            sources=[SourceFile(path="main.py", content=kernel_code)],
        )

        workloads = trace_set.workloads.get(task_id, [])
        if not workloads:
            return None

        output = flashinfer_bench_run_sanitizer(
            solution=solution,
            workload=workloads[0],
            sanitizer_types=["memcheck", "racecheck"],
            timeout=300,
        )

        # Check for errors in output
        if output and ("ERROR" in output.upper() or "INVALID" in output.upper()):
            return output
        return None

    except Exception as e:
        logger.warning(f"Sanitizer run failed: {e}")
        return None


def pack_submission(
    kernel_code: str,
    task_id: str,
    track: str,
    author: str = "kernforge",
    output_dir: str = "./submission",
) -> str:
    """
    Pack a kernel into the official submission format using flashinfer-bench.

    Returns path to the packed solution.json.
    """
    try:
        from flashinfer_bench import BuildSpec
        from flashinfer_bench.agents import pack_solution_from_files
    except ImportError:
        # Fallback: manual JSON packing
        return _manual_pack(kernel_code, task_id, track, author, output_dir)

    import os, tempfile, json

    # Write kernel to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_path = os.path.join(tmpdir, "main.py")
        with open(kernel_path, "w") as f:
            f.write(kernel_code)

        spec = BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point="main.py::run",
            dependencies=[],
            destination_passing_style=False,
        )

        solution = pack_solution_from_files(
            path=tmpdir,
            spec=spec,
            name=f"kernforge_{track}",
            definition=task_id,
            author=author,
        )

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "solution.json")
        with open(out_path, "w") as f:
            json.dump(solution.model_dump(), f, indent=2)

        return out_path


def _manual_pack(kernel_code, task_id, track, author, output_dir):
    """Fallback packing without flashinfer-bench."""
    import json, os

    solution = {
        "name": f"kernforge_{track}",
        "definition": task_id,
        "author": author,
        "spec": {
            "language": "triton",
            "target_hardware": ["cuda"],
            "entry_point": "main.py::run",
            "dependencies": [],
            "destination_passing_style": False,
        },
        "sources": [
            {"path": "main.py", "content": kernel_code}
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "solution.json")
    with open(out_path, "w") as f:
        json.dump(solution, f, indent=2)

    return out_path


def _parse_ncu_output(ncu_output: str) -> NCUProfile:
    """Parse NCU output text into structured profile."""
    if not ncu_output:
        return NCUProfile(bottleneck="unknown", raw_output="")

    profile = NCUProfile(bottleneck="unknown", raw_output=ncu_output)

    # Parse key metrics from NCU output
    import re

    # DRAM throughput
    dram_match = re.search(r"DRAM.*?(\d+\.?\d*)%", ncu_output, re.IGNORECASE)
    if dram_match:
        profile.dram_throughput_pct = float(dram_match.group(1))

    # SM throughput / compute utilization
    sm_match = re.search(r"(?:SM|Compute).*?(\d+\.?\d*)%", ncu_output, re.IGNORECASE)
    if sm_match:
        profile.sm_throughput_pct = float(sm_match.group(1))

    # Achieved occupancy
    occ_match = re.search(r"[Oo]ccupancy.*?(\d+\.?\d*)%", ncu_output)
    if occ_match:
        profile.achieved_occupancy_pct = float(occ_match.group(1))

    # Registers per thread
    reg_match = re.search(r"[Rr]egisters.*?(\d+)", ncu_output)
    if reg_match:
        profile.registers_per_thread = int(reg_match.group(1))

    # Local memory (spills)
    local_load = re.search(r"[Ll]ocal.*?[Ll]oad.*?(\d+)", ncu_output)
    if local_load:
        profile.local_load_bytes = int(local_load.group(1))

    local_store = re.search(r"[Ll]ocal.*?[Ss]tore.*?(\d+)", ncu_output)
    if local_store:
        profile.local_store_bytes = int(local_store.group(1))

    # Determine bottleneck
    has_spills = profile.local_load_bytes > 0 or profile.local_store_bytes > 0
    if has_spills:
        profile.bottleneck = "register_spills"
    elif profile.achieved_occupancy_pct > 0 and profile.achieved_occupancy_pct < 25:
        profile.bottleneck = "low_occupancy"
    elif profile.dram_throughput_pct > profile.sm_throughput_pct * 1.5:
        profile.bottleneck = "memory_bandwidth"
    elif profile.sm_throughput_pct > profile.dram_throughput_pct * 1.5:
        profile.bottleneck = "compute"
    elif profile.dram_throughput_pct > 50:
        profile.bottleneck = "memory_bandwidth"
    elif profile.sm_throughput_pct > 50:
        profile.bottleneck = "compute"

    return profile
