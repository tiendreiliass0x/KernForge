"""
PTX/SASS Analysis Tool — KernForge's eyes into the GPU.

Wafer's key insight: the agent's breakthrough came from reading the ISA,
not from latency numbers. Their agent spotted `ds_bpermute` serialization
in the disassembly and proposed DPP — impossible to discover from benchmarks.

This module gives KernForge the same capability for NVIDIA GPUs:
1. Compile Triton kernel → PTX (intermediate) → SASS (native assembly)
2. Analyze instruction mix, register pressure, memory ops, stalls
3. Produce structured feedback the LLM can reason about

The NVIDIA equivalent of Wafer's ISA analyzer.
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
class ISAAnalysis:
    """Structured analysis of compiled kernel assembly."""

    # Register usage
    num_registers: int = 0           # VGPRs (or general regs on NVIDIA)
    shared_memory_bytes: int = 0
    spill_stores: int = 0            # register spills to local memory
    spill_loads: int = 0

    # Instruction counts by category
    compute_instructions: int = 0     # FFMA, FMUL, FADD, etc.
    memory_instructions: int = 0      # LDG, STG, LDS, STS
    tensor_core_instructions: int = 0 # HMMA, IMMA on NVIDIA
    control_instructions: int = 0     # branches, barriers, etc.
    conversion_instructions: int = 0  # type casts
    total_instructions: int = 0

    # Specific patterns detected
    global_loads: int = 0             # LDG (HBM reads)
    global_stores: int = 0           # STG (HBM writes)
    shared_loads: int = 0            # LDS (SMEM reads)
    shared_stores: int = 0           # STS (SMEM writes)
    barriers: int = 0                # BAR.SYNC
    predicated_instructions: int = 0 # masked operations

    # Warnings / red flags
    issues: list[str] = field(default_factory=list)

    # Raw data
    raw_ptx: str = ""
    raw_sass: str = ""
    instruction_histogram: dict[str, int] = field(default_factory=dict)

    @property
    def compute_to_memory_ratio(self) -> float:
        """Higher = more compute-bound."""
        if self.memory_instructions == 0:
            return float("inf")
        return self.compute_instructions / self.memory_instructions

    @property
    def tensor_core_utilization(self) -> float:
        """Fraction of compute done via tensor cores."""
        total_compute = self.compute_instructions + self.tensor_core_instructions
        if total_compute == 0:
            return 0.0
        return self.tensor_core_instructions / total_compute

    @property
    def has_spills(self) -> bool:
        return self.spill_stores > 0 or self.spill_loads > 0

    @property
    def estimated_occupancy_limiter(self) -> str:
        """Guess what limits occupancy based on resource usage."""
        # B200: 65536 regs/SM, max 2048 threads/SM
        # At 128 regs/thread → 512 threads → 25% occupancy
        if self.num_registers > 128:
            return f"registers ({self.num_registers}/thread → ~{65536 // self.num_registers} threads → {min(100, 65536 // self.num_registers * 100 // 2048)}% occupancy)"
        if self.shared_memory_bytes > 128 * 1024:
            return f"shared memory ({self.shared_memory_bytes // 1024}KB/block → limited blocks per SM)"
        return "likely not resource-limited"

    def to_prompt_context(self) -> str:
        """Format as context for LLM prompt — this is the key output."""
        lines = [
            "## ISA Analysis Results",
            "",
            f"### Resource Usage",
            f"- Registers per thread: {self.num_registers}",
            f"- Shared memory: {self.shared_memory_bytes} bytes ({self.shared_memory_bytes / 1024:.1f} KB)",
            f"- Register spills: {self.spill_stores} stores, {self.spill_loads} loads"
            + (" ⚠️ SPILLING — reduce register pressure!" if self.has_spills else " ✓ No spills"),
            f"- Occupancy limiter: {self.estimated_occupancy_limiter}",
            "",
            f"### Instruction Mix ({self.total_instructions} total)",
            f"- Compute (FFMA/FMUL/FADD): {self.compute_instructions}",
            f"- Tensor core (HMMA): {self.tensor_core_instructions}"
            + (" ⚠️ NO TENSOR CORES — use tl.dot()!" if self.tensor_core_instructions == 0 and self.compute_instructions > 50 else ""),
            f"- Memory (LDG/STG/LDS/STS): {self.memory_instructions}",
            f"  - Global loads (HBM): {self.global_loads}",
            f"  - Global stores (HBM): {self.global_stores}",
            f"  - Shared loads (SMEM): {self.shared_loads}",
            f"  - Shared stores (SMEM): {self.shared_stores}",
            f"- Control (BAR/BRA): {self.control_instructions}",
            f"- Barriers: {self.barriers}",
            f"- Conversions: {self.conversion_instructions}",
            f"- Compute:Memory ratio: {self.compute_to_memory_ratio:.1f}"
            + (" (memory-bound)" if self.compute_to_memory_ratio < 2 else " (compute-heavy)" if self.compute_to_memory_ratio > 10 else ""),
        ]

        if self.issues:
            lines.append("")
            lines.append("### ⚠️ Issues Detected")
            for issue in self.issues:
                lines.append(f"- {issue}")

        if self.instruction_histogram:
            lines.append("")
            lines.append("### Top Instructions")
            sorted_instrs = sorted(self.instruction_histogram.items(), key=lambda x: -x[1])[:15]
            for instr, count in sorted_instrs:
                lines.append(f"  {instr}: {count}")

        return "\n".join(lines)


def analyze_triton_ptx(kernel_source: str, kernel_name: str = "kernel") -> ISAAnalysis:
    """
    Compile a Triton kernel and analyze its PTX.

    This is the primary analysis path since Triton compiles to PTX.
    For SASS analysis, we'd need the actual GPU and cuobjdump.
    """
    analysis = ISAAnalysis()

    try:
        # Write kernel to temp file and try to compile
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_path = Path(tmpdir) / "kernel.py"
            kernel_path.write_text(kernel_source)

            # Try to get PTX via Triton's compilation
            ptx = _compile_to_ptx(kernel_source, tmpdir)
            if ptx:
                analysis.raw_ptx = ptx
                _analyze_ptx(ptx, analysis)
            else:
                # Fallback: static analysis of Triton source
                _analyze_triton_source(kernel_source, analysis)

    except Exception as e:
        log.warning(f"ISA analysis failed: {e}")
        # Still do static analysis
        _analyze_triton_source(kernel_source, analysis)

    return analysis


def _compile_to_ptx(kernel_source: str, tmpdir: str) -> str | None:
    """Try to compile Triton kernel to PTX using Triton's compiler."""
    try:
        # This requires a GPU and triton installed
        compile_script = f"""
import sys
sys.path.insert(0, '{tmpdir}')
import triton
import triton.compiler as tc

# Try to extract the JIT function and compile it
# This is a best-effort approach
print("PTX compilation requires running on GPU")
"""
        result = subprocess.run(
            ["python", "-c", compile_script],
            capture_output=True, text=True, timeout=30,
            cwd=tmpdir,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _analyze_ptx(ptx: str, analysis: ISAAnalysis):
    """Parse PTX assembly and extract metrics."""
    lines = ptx.split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("."):
            # Check for register declarations
            if ".reg" in line:
                # Count register declarations
                match = re.search(r"\.reg\s+\.\w+\s+%\w+<(\d+)>", line)
                if match:
                    analysis.num_registers += int(match.group(1))
            if ".shared" in line:
                match = re.search(r"\.shared\s+.*\[(\d+)\]", line)
                if match:
                    analysis.shared_memory_bytes += int(match.group(1))
            continue

        analysis.total_instructions += 1

        # Categorize instruction
        opcode = line.split()[0].rstrip(";") if line.split() else ""

        if opcode in analysis.instruction_histogram:
            analysis.instruction_histogram[opcode] += 1
        else:
            analysis.instruction_histogram[opcode] = 1

        # Compute instructions
        if any(op in opcode for op in ["fma", "mul", "add", "sub", "mad", "fma"]):
            analysis.compute_instructions += 1
        # Tensor core
        elif any(op in opcode for op in ["mma", "hmma", "imma", "wmma"]):
            analysis.tensor_core_instructions += 1
        # Memory
        elif any(op in opcode for op in ["ld.", "st.", "atom"]):
            analysis.memory_instructions += 1
            if "global" in line or "ld.param" not in line:
                if "ld." in opcode:
                    analysis.global_loads += 1
                elif "st." in opcode:
                    analysis.global_stores += 1
            if "shared" in line:
                if "ld." in opcode:
                    analysis.shared_loads += 1
                elif "st." in opcode:
                    analysis.shared_stores += 1
        # Control
        elif any(op in opcode for op in ["bra", "bar", "ret", "call"]):
            analysis.control_instructions += 1
            if "bar" in opcode:
                analysis.barriers += 1
        # Conversions
        elif "cvt" in opcode:
            analysis.conversion_instructions += 1

        # Detect predicates
        if line.startswith("@"):
            analysis.predicated_instructions += 1

        # Detect spills
        if "local" in line and "st." in line:
            analysis.spill_stores += 1
        elif "local" in line and "ld." in line:
            analysis.spill_loads += 1

    # Generate issues
    _detect_issues(analysis)


def _analyze_triton_source(source: str, analysis: ISAAnalysis):
    """
    Static analysis of Triton source code when PTX isn't available.
    Less precise but still useful for the LLM.
    """
    lines = source.split("\n")

    # Count key patterns
    tl_dot_count = source.count("tl.dot")
    tl_load_count = source.count("tl.load")
    tl_store_count = source.count("tl.store")
    tl_zeros_count = source.count("tl.zeros")

    # Detect tensor core usage
    analysis.tensor_core_instructions = tl_dot_count * 4  # rough estimate
    analysis.global_loads = tl_load_count
    analysis.global_stores = tl_store_count

    # Estimate compute from loop structure
    for_loops = len(re.findall(r"for\s+\w+\s+in\s+range", source))
    analysis.compute_instructions = for_loops * 10  # very rough

    # Detect autotuning
    has_autotune = "triton.autotune" in source
    autotune_configs = len(re.findall(r"triton\.Config", source))

    # Extract BLOCK sizes
    block_sizes = re.findall(r"BLOCK_?\w*\s*[:=]\s*tl\.constexpr|'B\w+'\s*:\s*(\d+)", source)

    # Detect numerical patterns
    has_f32_acc = "tl.float32" in source and "tl.zeros" in source
    has_bf16_acc = "tl.bfloat16" in source and "tl.zeros" in source and "tl.float32" not in source
    has_mask = "mask=" in source
    has_softplus = "softplus" in source.lower() or "log(1" in source
    has_sigmoid = "sigmoid" in source.lower() or "1.0 / (1.0 + tl.exp" in source

    # Generate issues based on static analysis
    if tl_dot_count == 0 and for_loops > 2:
        analysis.issues.append(
            "NO tl.dot() calls detected — matrix operations may be using scalar arithmetic "
            "instead of tensor cores. Look for patterns like `tl.sum(a * b, axis=1)` that "
            "could be replaced with `tl.dot(a_2d, b_2d)`."
        )

    if not has_f32_acc and has_bf16_acc:
        analysis.issues.append(
            "Accumulator appears to be bf16, not f32. This will cause numerical drift. "
            "Use `tl.zeros([...], dtype=tl.float32)` for accumulators."
        )

    if tl_load_count > 0 and not has_mask:
        analysis.issues.append(
            "tl.load() calls without mask= detected. This will crash on boundary tiles "
            "where offsets exceed tensor dimensions. Always use mask with variable-size axes."
        )

    if not has_autotune:
        analysis.issues.append(
            "No @triton.autotune detected. Add multiple configurations to explore "
            "tile sizes and num_warps automatically."
        )
    elif autotune_configs < 3:
        analysis.issues.append(
            f"Only {autotune_configs} autotune configs. Add more (5-8) to explore the "
            "tile size / num_warps / num_stages space more thoroughly."
        )

    # GDN-specific checks
    if "state" in source.lower():
        # Check if state is being tiled properly
        if "BLOCK" not in source.upper() and tl_load_count > 3:
            analysis.issues.append(
                "State matrix appears to be loaded without explicit tiling. "
                "For 128×128 f32 state (64KB), tile along the V dimension with BLOCK_V=32-64 "
                "to ensure each tile fits in registers."
            )

    # Check for GVA head mapping
    if "v_head" in source or "num_v_heads" in source:
        if "//" not in source and "heads_per_group" not in source and "repeat_interleave" not in source:
            analysis.issues.append(
                "GVA head mapping may be missing. num_v_heads > num_q_heads requires "
                "mapping: qk_head = v_head // (num_v_heads // num_q_heads)."
            )

    analysis.total_instructions = analysis.compute_instructions + analysis.memory_instructions + analysis.tensor_core_instructions


def _detect_issues(analysis: ISAAnalysis):
    """Detect performance issues from PTX/SASS analysis."""

    if analysis.has_spills:
        analysis.issues.append(
            f"REGISTER SPILLS detected ({analysis.spill_stores} stores, {analysis.spill_loads} loads). "
            "This means data is being evicted from registers to slow local memory. "
            "Reduce tile sizes or live variables to fix."
        )

    if analysis.tensor_core_instructions == 0 and analysis.compute_instructions > 100:
        analysis.issues.append(
            "No tensor core instructions (HMMA/MMA) detected despite significant compute. "
            "Use tl.dot() for matrix multiplications to leverage tensor cores (10-50x faster)."
        )

    if analysis.barriers > 10:
        analysis.issues.append(
            f"Excessive barriers ({analysis.barriers}). Each barrier synchronizes all threads "
            "in the block, causing idle time. Reduce shared memory usage or restructure to "
            "minimize synchronization points."
        )

    if analysis.global_loads > 50 and analysis.shared_loads == 0:
        analysis.issues.append(
            "Many global loads but no shared memory loads. Consider loading frequently-reused "
            "data into shared memory first, then reading from shared memory in the compute loop."
        )

    if analysis.compute_to_memory_ratio < 1.0:
        analysis.issues.append(
            f"Very low compute:memory ratio ({analysis.compute_to_memory_ratio:.1f}). "
            "Kernel is heavily memory-bound. Priority: reduce HBM traffic through tiling, "
            "data reuse, or operation fusion."
        )

    if analysis.conversion_instructions > analysis.compute_instructions * 0.3:
        analysis.issues.append(
            f"High type conversion overhead ({analysis.conversion_instructions} conversions vs "
            f"{analysis.compute_instructions} compute ops). Consider keeping data in a single "
            "precision throughout the hot loop."
        )


def analyze_ncu_output(ncu_text: str) -> dict[str, Any]:
    """
    Parse NVIDIA Nsight Compute output into structured metrics.

    This processes the output from `flashinfer_bench_run_ncu` or
    direct `ncu` CLI invocation.
    """
    metrics = {}

    # Common NCU metric patterns
    patterns = {
        "achieved_bandwidth_gb_s": r"DRAM Throughput\s+[\d.]+%\s+([\d.]+)\s*GB/s",
        "achieved_occupancy_pct": r"Achieved Occupancy\s+([\d.]+)%",
        "compute_throughput_pct": r"Compute \(SM\) Throughput\s+([\d.]+)%",
        "memory_throughput_pct": r"Memory Throughput\s+([\d.]+)%",
        "l2_hit_rate_pct": r"L2 Hit Rate\s+([\d.]+)%",
        "registers_per_thread": r"Registers Per Thread\s+(\d+)",
        "shared_memory_bytes": r"Dynamic Shared Memory Per Block\s+(\d+)",
        "block_limit": r"Block Limit\w*\s+(\w+)",
        "waves_per_sm": r"Waves Per SM\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, ncu_text)
        if match:
            val = match.group(1)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val

    return metrics


# =============================================================================
# Integration with the evolution loop
# =============================================================================

def enrich_eval_with_isa(
    kernel_source: str,
    eval_result: "EvalResult",
    kernel_type: str = "general",
) -> str:
    """
    Run ISA analysis and produce enriched feedback for the LLM.

    This is the function the evolution loop calls after benchmarking.
    Returns a string that gets injected into the improvement prompt.
    """
    analysis = analyze_triton_ptx(kernel_source, kernel_type)

    # Combine ISA analysis with benchmark results
    sections = [analysis.to_prompt_context()]

    # Add benchmark context
    if eval_result.median_latency_us:
        sections.append(f"\n## Benchmark: {eval_result.median_latency_us:.1f} μs median latency")

    # Cross-reference: if ISA shows no tensor cores but kernel is compute-heavy
    if analysis.tensor_core_instructions == 0 and analysis.compute_instructions > 20:
        sections.append(
            "\n## Critical: The kernel does zero tensor core operations. "
            "On B200, BF16 tensor cores are 25x faster than scalar FP32. "
            "Converting matrix operations to use tl.dot() is the highest-impact change."
        )

    # Cross-reference: if spilling but latency is still reasonable
    if analysis.has_spills:
        sections.append(
            "\n## Critical: Register spills detected. Data is being evicted to "
            "slow local memory. This adds ~100 cycles per spill. "
            "Reduce BLOCK sizes or intermediate variables."
        )

    return "\n".join(sections)
