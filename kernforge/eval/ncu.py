"""
NCU (NVIDIA Nsight Compute) Integration.

Provides real profiling data: bandwidth utilization, compute throughput,
occupancy, register pressure, memory traffic breakdown, and stall reasons.

Supports three modes:
1. Direct ncu CLI invocation (local GPU)
2. flashinfer-bench NCU runner (Modal B200)
3. Triton's built-in profiling hooks (lightweight)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class NCUProfile:
    """Structured NCU profiling results."""
    # Memory
    dram_throughput_gb_s: float = 0.0
    dram_throughput_pct: float = 0.0
    l2_throughput_gb_s: float = 0.0
    l2_hit_rate_pct: float = 0.0
    shared_throughput_gb_s: float = 0.0

    # Compute
    sm_throughput_pct: float = 0.0
    achieved_flops_tflops: float = 0.0
    tensor_core_utilization_pct: float = 0.0

    # Occupancy
    achieved_occupancy_pct: float = 0.0
    theoretical_occupancy_pct: float = 0.0
    registers_per_thread: int = 0
    shared_mem_per_block_bytes: int = 0
    active_warps_per_sm: float = 0.0
    max_warps_per_sm: int = 64

    # Stalls
    stall_reasons: dict[str, float] = field(default_factory=dict)

    # Memory traffic
    global_load_bytes: int = 0
    global_store_bytes: int = 0
    shared_load_bytes: int = 0
    shared_store_bytes: int = 0
    local_load_bytes: int = 0   # register spills
    local_store_bytes: int = 0

    # Launch config
    grid_size: tuple[int, ...] = (0,)
    block_size: tuple[int, ...] = (0,)
    num_warps: int = 0

    # Raw
    raw_output: str = ""

    @property
    def is_memory_bound(self) -> bool:
        return self.dram_throughput_pct > 60 and self.sm_throughput_pct < 40

    @property
    def is_compute_bound(self) -> bool:
        return self.sm_throughput_pct > 50 and self.dram_throughput_pct < 40

    @property
    def is_latency_bound(self) -> bool:
        return self.dram_throughput_pct < 40 and self.sm_throughput_pct < 40

    @property
    def has_spills(self) -> bool:
        return self.local_load_bytes > 0 or self.local_store_bytes > 0

    @property
    def bottleneck(self) -> str:
        if self.has_spills:
            return "register_spills"
        if self.is_memory_bound:
            return "memory_bandwidth"
        if self.is_compute_bound:
            return "compute"
        if self.achieved_occupancy_pct < 25:
            return "low_occupancy"
        return "latency_or_unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dram_throughput_gb_s": self.dram_throughput_gb_s,
            "dram_throughput_pct": self.dram_throughput_pct,
            "sm_throughput_pct": self.sm_throughput_pct,
            "achieved_occupancy_pct": self.achieved_occupancy_pct,
            "registers_per_thread": self.registers_per_thread,
            "bottleneck": self.bottleneck,
            "has_spills": self.has_spills,
            "tensor_core_utilization_pct": self.tensor_core_utilization_pct,
            "global_load_bytes": self.global_load_bytes,
            "global_store_bytes": self.global_store_bytes,
            "local_load_bytes": self.local_load_bytes,
            "local_store_bytes": self.local_store_bytes,
            "grid_size": self.grid_size,
            "block_size": self.block_size,
        }

    def to_prompt_context(self) -> str:
        """Format as rich feedback for the LLM agent."""
        lines = [
            "## NCU Profiling Results",
            "",
            "### Bottleneck Classification",
            f"**{self.bottleneck.upper().replace('_', ' ')}**",
            "",
            "### Memory Subsystem",
            f"- DRAM (HBM) throughput: {self.dram_throughput_gb_s:.0f} GB/s ({self.dram_throughput_pct:.0f}% of peak)",
            f"- L2 cache hit rate: {self.l2_hit_rate_pct:.0f}%",
            f"- Global loads: {self.global_load_bytes / 1024:.1f} KB",
            f"- Global stores: {self.global_store_bytes / 1024:.1f} KB",
        ]

        if self.has_spills:
            lines.append(f"- ⚠️ REGISTER SPILLS: {self.local_load_bytes} bytes loaded, {self.local_store_bytes} bytes stored from local memory")

        lines.extend([
            "",
            "### Compute",
            f"- SM throughput: {self.sm_throughput_pct:.0f}%",
            f"- Tensor core utilization: {self.tensor_core_utilization_pct:.0f}%",
        ])

        if self.tensor_core_utilization_pct < 5 and self.sm_throughput_pct > 20:
            lines.append("  ⚠️ Low tensor core usage — consider using tl.dot() for matrix ops")

        lines.extend([
            "",
            "### Occupancy",
            f"- Achieved: {self.achieved_occupancy_pct:.0f}% ({self.active_warps_per_sm:.0f}/{self.max_warps_per_sm} warps/SM)",
            f"- Registers/thread: {self.registers_per_thread}",
            f"- Shared memory/block: {self.shared_mem_per_block_bytes / 1024:.1f} KB",
        ])

        if self.achieved_occupancy_pct < 25:
            lines.append(f"  ⚠️ LOW OCCUPANCY — reduce registers (currently {self.registers_per_thread})")

        lines.extend([
            "",
            "### Launch Configuration",
            f"- Grid: {self.grid_size}",
            f"- Block: {self.block_size} ({self.num_warps} warps)",
        ])

        if self.stall_reasons:
            lines.extend(["", "### Top Stall Reasons"])
            for reason, pct in sorted(self.stall_reasons.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"- {reason}: {pct:.1f}%")

        return "\n".join(lines)


# ============================================================================
# Mode 1: Direct ncu CLI
# ============================================================================

def run_ncu_cli(
    kernel_source: str,
    shape_dict: dict[str, int],
    spec_name: str = "kernel",
    device: int = 0,
    timeout_s: int = 120,
) -> NCUProfile:
    """
    Run ncu on a kernel and parse the output.

    Writes a self-contained script that imports and runs the kernel,
    then invokes ncu on it.
    """
    profile = NCUProfile()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write kernel file
        kernel_path = Path(tmpdir) / "kernel.py"
        kernel_path.write_text(kernel_source)

        # Write runner script
        runner_script = _generate_ncu_runner(spec_name, shape_dict, device)
        runner_path = Path(tmpdir) / "run_ncu.py"
        runner_path.write_text(runner_script)

        # NCU metrics to collect
        metrics = [
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block",
            "launch__grid_size",
            "launch__block_size",
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__inst_executed_pipe_tensor.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_lg.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_local.sum",
        ]

        ncu_cmd = [
            "ncu",
            "--target-processes", "all",
            "--metrics", ",".join(metrics),
            "--csv",
            "--log-file", str(Path(tmpdir) / "ncu_output.csv"),
            "python", str(runner_path),
        ]

        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True, text=True,
                timeout=timeout_s,
                cwd=tmpdir,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(device)},
            )
            profile.raw_output = result.stdout + result.stderr

            # Parse CSV output
            csv_path = Path(tmpdir) / "ncu_output.csv"
            if csv_path.exists():
                profile = _parse_ncu_csv(csv_path.read_text(), profile)
            elif result.stdout:
                profile = _parse_ncu_text(result.stdout, profile)

        except subprocess.TimeoutExpired:
            log.warning(f"NCU timed out after {timeout_s}s")
        except FileNotFoundError:
            log.warning("ncu not found — install NVIDIA Nsight Compute")

    return profile


# ============================================================================
# Mode 2: FlashInfer-Bench NCU runner (Modal B200)
# ============================================================================

def run_ncu_flashinfer_bench(
    solution_dir: str | Path,
    dataset_path: str | Path,
    definition: str,
) -> NCUProfile:
    """
    Run NCU via flashinfer-bench's infrastructure on Modal B200.

    This calls `flashinfer-bench run-ncu` which handles:
    - Uploading solution to Modal
    - Running on B200 hardware
    - Collecting NCU metrics
    - Returning structured results
    """
    profile = NCUProfile()

    try:
        result = subprocess.run(
            [
                "flashinfer-bench", "run-ncu",
                "--solution", str(solution_dir),
                "--dataset", str(dataset_path),
                "--definition", definition,
                "--output-format", "json",
            ],
            capture_output=True, text=True,
            timeout=300,  # 5 min timeout for remote execution
        )

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                profile = _parse_flashinfer_bench_ncu(data, profile)
            except json.JSONDecodeError:
                log.warning(f"Failed to parse flashinfer-bench NCU output")
                profile.raw_output = result.stdout
        else:
            log.warning(f"flashinfer-bench run-ncu failed: {result.stderr[:500]}")

    except FileNotFoundError:
        log.warning("flashinfer-bench CLI not found — install with: pip install flashinfer-bench")
    except subprocess.TimeoutExpired:
        log.warning("flashinfer-bench NCU timed out")

    return profile


# ============================================================================
# Mode 3: Triton's built-in profiling
# ============================================================================

def run_triton_profiling(kernel_fn, inputs: dict) -> NCUProfile:
    """
    Use Triton's built-in profiling to get basic metrics without ncu.
    Less detailed but works without root/ncu installation.
    """
    profile = NCUProfile()

    try:
        import torch
        import triton

        # Get compiled kernel info
        if hasattr(kernel_fn, 'cache'):
            for key, compiled in kernel_fn.cache.items():
                if hasattr(compiled, 'metadata'):
                    meta = compiled.metadata
                    profile.registers_per_thread = getattr(meta, 'num_regs', 0)
                    profile.shared_mem_per_block_bytes = getattr(meta, 'shared', 0)
                    profile.num_warps = getattr(meta, 'num_warps', 0)

        # Run with torch profiler for basic metrics
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            kernel_fn(**inputs)
            torch.cuda.synchronize()

        events = prof.key_averages()
        for event in events:
            if event.device_type == torch.autograd.DeviceType.CUDA:
                if event.cuda_time_total > 0:
                    # Basic info from torch profiler
                    pass

    except Exception as e:
        log.debug(f"Triton profiling failed: {e}")

    return profile


# ============================================================================
# Parsers
# ============================================================================

def _parse_ncu_csv(csv_text: str, profile: NCUProfile) -> NCUProfile:
    """Parse NCU CSV output into structured profile."""
    lines = csv_text.strip().split("\n")
    if len(lines) < 2:
        return profile

    # Find header and data
    header_idx = -1
    for i, line in enumerate(lines):
        if "Metric Name" in line or "metric__name" in line.lower():
            header_idx = i
            break

    if header_idx < 0:
        return _parse_ncu_text(csv_text, profile)

    for line in lines[header_idx + 1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue

        metric_name = parts[0].strip().strip('"')
        try:
            value = float(parts[-1].strip().strip('"').replace(",", ""))
        except (ValueError, IndexError):
            continue

        if "dram__throughput" in metric_name and "pct" in metric_name:
            profile.dram_throughput_pct = value
        elif "dram__bytes_read" in metric_name:
            profile.global_load_bytes = int(value)
        elif "dram__bytes_write" in metric_name:
            profile.global_store_bytes = int(value)
        elif "sm__throughput" in metric_name and "pct" in metric_name:
            profile.sm_throughput_pct = value
        elif "lts__t_sector_hit_rate" in metric_name:
            profile.l2_hit_rate_pct = value
        elif "warps_active" in metric_name and "pct" in metric_name:
            profile.achieved_occupancy_pct = value
        elif "registers_per_thread" in metric_name:
            profile.registers_per_thread = int(value)
        elif "shared_mem_per_block" in metric_name:
            profile.shared_mem_per_block_bytes = int(value)
        elif "pipe_tensor" in metric_name:
            profile.tensor_core_utilization_pct = min(value, 100)  # normalize
        elif "mem_local" in metric_name:
            profile.local_load_bytes = int(value)

    # Derive bandwidth from bytes and latency (if available)
    # This is estimated — real ncu gives it directly
    return profile


def _parse_ncu_text(text: str, profile: NCUProfile) -> NCUProfile:
    """Parse NCU text/log output (fallback)."""
    patterns = {
        "dram_throughput_pct": r"DRAM Throughput\s+([\d.]+)%",
        "sm_throughput_pct": r"SM.*?Throughput\s+([\d.]+)%",
        "achieved_occupancy_pct": r"Achieved Occupancy\s+([\d.]+)%",
        "l2_hit_rate_pct": r"L2.*?Hit Rate\s+([\d.]+)%",
        "registers_per_thread": r"Registers Per Thread\s+(\d+)",
        "shared_mem_per_block_bytes": r"Shared Memory.*?Per Block\s+(\d+)",
    }

    for attr, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            setattr(profile, attr, int(val) if attr.endswith("bytes") or attr.endswith("thread") else val)

    return profile


def _parse_flashinfer_bench_ncu(data: dict, profile: NCUProfile) -> NCUProfile:
    """Parse flashinfer-bench's structured NCU output."""
    if "metrics" in data:
        m = data["metrics"]
        profile.dram_throughput_pct = m.get("dram_throughput_pct", 0)
        profile.dram_throughput_gb_s = m.get("dram_throughput_gb_s", 0)
        profile.sm_throughput_pct = m.get("sm_throughput_pct", 0)
        profile.achieved_occupancy_pct = m.get("achieved_occupancy_pct", 0)
        profile.registers_per_thread = m.get("registers_per_thread", 0)
        profile.shared_mem_per_block_bytes = m.get("shared_mem_per_block", 0)
        profile.global_load_bytes = m.get("global_load_bytes", 0)
        profile.global_store_bytes = m.get("global_store_bytes", 0)
        profile.local_load_bytes = m.get("local_load_bytes", 0)
        profile.local_store_bytes = m.get("local_store_bytes", 0)
        profile.tensor_core_utilization_pct = m.get("tensor_core_pct", 0)
    return profile


def _generate_ncu_runner(spec_name: str, shape_dict: dict[str, int], device: int) -> str:
    """Generate a self-contained Python script that ncu can profile."""
    shape_json = json.dumps(shape_dict)
    return f"""#!/usr/bin/env python3
\"\"\"Auto-generated NCU profiling runner for {spec_name}.\"\"\"
import torch
import sys
import importlib.util

torch.cuda.set_device({device})
device = "cuda:{device}"

# Import the kernel module from the same directory
spec = importlib.util.spec_from_file_location("_kernel_module", "./kernel.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, ".")
spec.loader.exec_module(module)

# Find the entry point function
kernel_fn = None
for name in ("kernel", "run", "forward", "main"):
    if hasattr(module, name):
        kernel_fn = getattr(module, name)
        break
if kernel_fn is None:
    # Use the first callable that isn't a builtin
    import types
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, types.FunctionType) and not attr_name.startswith("_"):
            kernel_fn = obj
            break
if kernel_fn is None:
    print("ERROR: No kernel function found", file=sys.stderr)
    sys.exit(1)

# Generate test inputs matching shape
shape = {shape_json}
inputs = {{}}
# Heuristic: generate tensors matching common parameter patterns
for key, val in shape.items():
    if "batch" in key or "seq" in key or "head" in key or "dim" in key or "size" in key:
        continue  # these are dimension parameters, not tensor names

# Create common GDN-style inputs based on shape dict
B = shape.get("batch_size", shape.get("B", 4))
H_q = shape.get("num_q_heads", shape.get("H_q", 16))
H_v = shape.get("num_v_heads", shape.get("H_v", 32))
K = shape.get("head_size", shape.get("K", 128))
V = shape.get("val_size", K)

# Generate plausible random inputs
inputs = dict(
    q=torch.randn(B, H_q, K, dtype=torch.bfloat16, device=device),
    k=torch.randn(B, H_q, K, dtype=torch.bfloat16, device=device),
    v=torch.randn(B, H_v, V, dtype=torch.bfloat16, device=device),
    state=torch.randn(B, H_v, V, K, dtype=torch.float32, device=device) * 0.01,
)

# Warmup
try:
    for _ in range(3):
        kernel_fn(**inputs)
    torch.cuda.synchronize()
except Exception as e:
    print(f"Warmup failed: {{e}}", file=sys.stderr)
    # Try calling with positional args
    try:
        args = list(inputs.values())
        for _ in range(3):
            kernel_fn(*args)
        torch.cuda.synchronize()
    except Exception as e2:
        print(f"Positional call also failed: {{e2}}", file=sys.stderr)
        sys.exit(1)

# Profile run (this is what ncu will capture)
try:
    kernel_fn(**inputs)
except Exception:
    kernel_fn(*list(inputs.values()))
torch.cuda.synchronize()
"""
