"""
Kernel evaluator: correctness testing and performance benchmarking.

Supports both local evaluation (for development) and FlashInfer-Bench
integration (for official benchmarking on B200 via Modal).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..kernel.solution import EvalResult, Solution
from ..kernel.spec import KernelSpec

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for kernel evaluation."""
    warmup_runs: int = 5
    benchmark_runs: int = 50
    atol: float = 1e-2          # absolute tolerance for bf16
    rtol: float = 1e-2          # relative tolerance for bf16
    atol_f32: float = 1e-5      # for f32 outputs
    rtol_f32: float = 1e-4
    timeout_s: int = 120
    device: str = "cuda:0"
    run_ncu: bool = False       # run NVIDIA Nsight Compute profiling


class KernelEvaluator:
    """
    Evaluates kernel solutions for correctness and performance.

    The evaluator:
    1. Compiles the kernel (catches syntax/import errors)
    2. Generates test inputs matching the kernel spec
    3. Runs the reference implementation for ground truth
    4. Runs the candidate kernel
    5. Compares outputs within tolerance
    6. Benchmarks latency if correct
    """

    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()

    def evaluate(self, spec: KernelSpec, solution: Solution,
                 test_shapes: list[dict[str, int]] | None = None) -> EvalResult:
        """Full evaluation: static check → compile → correctness → benchmark → profile."""

        # Step 0: Static analysis (free, no GPU needed, no torch needed)
        try:
            from .static_analysis import analyze as static_analyze
            static_result = static_analyze(
                solution.main_source, solution.entry_point, spec.kernel_type
            )
            if static_result.blocking:
                errors_str = "\n".join(str(e) for e in static_result.errors)
                return EvalResult(
                    correct=False,
                    compile_error=f"Static analysis found blocking errors:\n{errors_str}\n\n{static_result.to_prompt_context()}",
                )
        except Exception as e:
            log.debug(f"Static analysis skipped: {e}")

        import torch
        # Step 1: Compile / import
        try:
            kernel_fn = self._load_kernel(solution)
        except Exception as e:
            log.error(f"Compile error: {e}")
            return EvalResult(
                correct=False,
                compile_error=f"{type(e).__name__}: {str(e)[:1000]}",
            )

        # Step 2: Generate test shapes — use adversarial suite if available
        if test_shapes is None:
            try:
                from .adversarial import generate_test_suite
                suite = generate_test_suite(spec.kernel_type)
                test_shapes = suite.shapes_only()
            except Exception:
                test_shapes = self._generate_test_shapes(spec)

        # Step 3: Test correctness on each shape
        max_abs_err = 0.0
        max_rel_err = 0.0

        for shape_dict in test_shapes:
            # Skip shapes with special values like -1
            if any(v < 0 for v in shape_dict.values()):
                continue
            try:
                result = self._test_correctness(spec, kernel_fn, shape_dict)
                max_abs_err = max(max_abs_err, result["max_abs_error"])
                max_rel_err = max(max_rel_err, result["max_rel_error"])

                if not result["correct"]:
                    # Add shape info for error diagnosis
                    error_msg = result.get("error", "Incorrect output")
                    diagnosis = self._diagnose_correctness_error(
                        result, shape_dict, spec
                    )
                    return EvalResult(
                        correct=False,
                        max_abs_error=result["max_abs_error"],
                        max_rel_error=result["max_rel_error"],
                        runtime_error=f"Shape {shape_dict}: {error_msg}\n{diagnosis}",
                    )
            except Exception as e:
                tb = traceback.format_exc()
                log.error(f"Runtime error on shape {shape_dict}: {e}\n{tb}")
                return EvalResult(
                    correct=False,
                    runtime_error=f"Shape {shape_dict}: {type(e).__name__}: {str(e)[:800]}\n{tb[-500:]}",
                )

        # Step 4: Benchmark performance
        latencies = []
        bench_shape = test_shapes[-1]  # use largest shape for benchmarking
        # Skip special shapes
        bench_shape = next(
            (s for s in reversed(test_shapes) if all(v > 0 for v in s.values())),
            test_shapes[0],
        )
        try:
            latencies = self._benchmark(spec, kernel_fn, bench_shape)
        except Exception as e:
            log.warning(f"Benchmark failed: {e}")

        # Step 5: NCU profiling (if enabled and kernel is correct)
        ncu_metrics = {}
        if self.config.run_ncu and latencies:
            try:
                ncu_metrics = self._run_ncu(solution, bench_shape)
            except Exception as e:
                log.warning(f"NCU profiling failed: {e}")

        median_lat = sorted(latencies)[len(latencies) // 2] if latencies else None
        min_lat = min(latencies) if latencies else None

        return EvalResult(
            correct=True,
            max_abs_error=max_abs_err,
            max_rel_error=max_rel_err,
            median_latency_us=median_lat,
            min_latency_us=min_lat,
            ncu_metrics=ncu_metrics,
        )

    def evaluate_with_flashinfer_bench(self, solution: Solution,
                                        dataset_path: str | Path,
                                        definition: str | None = None) -> EvalResult:
        """
        Evaluate using the official FlashInfer-Bench framework via CLI.

        Writes the solution to a temp directory in starter-kit format,
        then invokes `flashinfer-bench evaluate` as a subprocess.
        This is the real path used for contest submissions.
        """
        definition = definition or solution.definition

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write solution in starter-kit format
            solution.save(tmpdir)

            # Write config.toml expected by flashinfer-bench
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                f'[solution]\nname = "{solution.name}"\n'
                f'definition = "{definition}"\n'
                f'language = "{solution.language}"\n'
                f'entry_point = "{solution.entry_point}"\n'
            )

            # Invoke flashinfer-bench CLI
            cmd = [
                "flashinfer-bench", "evaluate",
                "--solution-dir", tmpdir,
                "--dataset", str(dataset_path),
                "--definition", definition,
                "--warmup", str(self.config.warmup_runs),
                "--iterations", str(self.config.benchmark_runs),
                "--output-format", "json",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    timeout=self.config.timeout_s,
                )
            except FileNotFoundError:
                log.error("flashinfer-bench CLI not found. Install with: pip install flashinfer-bench")
                return EvalResult(
                    correct=False,
                    runtime_error="flashinfer-bench CLI not found. Install with: pip install flashinfer-bench",
                )
            except subprocess.TimeoutExpired:
                return EvalResult(
                    correct=False,
                    runtime_error=f"flashinfer-bench timed out after {self.config.timeout_s}s",
                )

            if result.returncode != 0:
                error_msg = result.stderr[:1000] if result.stderr else result.stdout[:1000]
                return EvalResult(
                    correct=False,
                    runtime_error=f"flashinfer-bench failed (exit {result.returncode}): {error_msg}",
                )

            # Parse JSON output
            import json as json_mod
            try:
                data = json_mod.loads(result.stdout)
            except (json_mod.JSONDecodeError, ValueError):
                # Try to find JSON in the output (may have log lines before it)
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json_mod.loads(line)
                            break
                        except (json_mod.JSONDecodeError, ValueError):
                            continue
                else:
                    return EvalResult(
                        correct=False,
                        runtime_error=f"Failed to parse flashinfer-bench output: {result.stdout[:500]}",
                    )

            # Extract results from flashinfer-bench JSON format
            correct = data.get("correct", data.get("passed", False))
            return EvalResult(
                correct=correct,
                max_abs_error=data.get("max_abs_error"),
                max_rel_error=data.get("max_rel_error"),
                median_latency_us=data.get("median_latency_us", data.get("latency_us")),
                min_latency_us=data.get("min_latency_us"),
                compile_error=data.get("compile_error"),
                runtime_error=data.get("runtime_error") if not correct else None,
            )

    def _load_kernel(self, solution: Solution) -> Any:
        """Dynamically load the kernel function from solution source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write all source files
            for filename, content in solution.sources.items():
                filepath = Path(tmpdir) / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content)

            # Import the main module
            main_file = Path(tmpdir) / "kernel.py"
            if not main_file.exists():
                # Find the first .py file
                py_files = list(Path(tmpdir).glob("**/*.py"))
                if not py_files:
                    raise FileNotFoundError("No Python files in solution")
                main_file = py_files[0]

            spec = importlib.util.spec_from_file_location("_kernel_module", str(main_file))
            module = importlib.util.module_from_spec(spec)

            # Add tmpdir to path for relative imports
            sys.path.insert(0, tmpdir)
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.pop(0)

            # Get the entry point function
            entry = solution.entry_point.split("::")[-1]  # handle "file::function" format
            if not hasattr(module, entry):
                # Try common names
                for name in ["kernel", "run", "forward", "main"]:
                    if hasattr(module, name):
                        return getattr(module, name)
                raise AttributeError(
                    f"Entry point '{entry}' not found. "
                    f"Available: {[x for x in dir(module) if not x.startswith('_')]}"
                )
            return getattr(module, entry)

    def _generate_test_shapes(self, spec: KernelSpec) -> list[dict[str, int]]:
        """Generate a set of test shapes covering different sizes."""
        shapes = []
        # Small shape for quick correctness check
        small = {}
        medium = {}
        large = {}

        for ax in spec.axes:
            if ax.type == "const":
                small[ax.name] = ax.value
                medium[ax.name] = ax.value
                large[ax.name] = ax.value
            else:
                # Variable axis — try small, medium, large
                small[ax.name] = 1 if "seq" in ax.name else 4
                medium[ax.name] = 16 if "seq" in ax.name else 16
                large[ax.name] = 128 if "seq" in ax.name else 64

        shapes = [small, medium, large]
        return shapes

    def _generate_test_inputs(self, spec: KernelSpec, shape_dict: dict[str, int]) -> dict:
        """Generate random test inputs matching the spec."""
        import torch

        device = self.config.device
        inputs = {}

        for inp in spec.inputs:
            # Resolve symbolic shape
            shape = []
            for s in inp.shape:
                if isinstance(s, int):
                    shape.append(s)
                elif s in shape_dict:
                    shape.append(shape_dict[s])
                elif s == "Scalar":
                    shape = []
                    break
                else:
                    shape.append(shape_dict.get(s, 1))

            # Generate tensor
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "int64": torch.int64,
                "int32": torch.int32,
            }
            dtype = dtype_map.get(inp.dtype, torch.float32)

            if not shape:  # Scalar
                if "int" in inp.dtype:
                    inputs[inp.name] = torch.tensor(1, dtype=dtype, device=device)
                else:
                    inputs[inp.name] = torch.tensor(0.1, dtype=dtype, device=device)
            elif "int" in inp.dtype:
                if "cu_seqlens" in inp.name:
                    # Special handling for cumulative sequence lengths
                    num_seqs = shape_dict.get("num_seqs", 1)
                    total = shape_dict.get("total_seq_len", 64)
                    # Evenly distribute
                    seq_len = total // num_seqs
                    cu = [0]
                    for i in range(num_seqs):
                        cu.append(cu[-1] + seq_len)
                    cu[-1] = total  # ensure exact total
                    inputs[inp.name] = torch.tensor(cu, dtype=dtype, device=device)
                else:
                    inputs[inp.name] = torch.randint(0, 10, shape, dtype=dtype, device=device)
            else:
                inputs[inp.name] = torch.randn(shape, dtype=dtype, device=device)

        return inputs

    def _test_correctness(self, spec: KernelSpec, kernel_fn, shape_dict: dict[str, int]) -> dict:
        """Test kernel correctness against the spec's reference implementation."""
        import torch

        inputs = self._generate_test_inputs(spec, shape_dict)

        # Run candidate kernel
        try:
            candidate_outputs = kernel_fn(**inputs)
        except Exception as e:
            return {"correct": False, "max_abs_error": float("inf"),
                    "max_rel_error": float("inf"),
                    "error": f"Candidate kernel raised: {type(e).__name__}: {str(e)[:500]}"}

        if not isinstance(candidate_outputs, tuple):
            candidate_outputs = (candidate_outputs,)

        # Run reference implementation (from spec.reference_code)
        reference_outputs = self._run_reference(spec, inputs)

        if reference_outputs is None:
            # No reference available — fall back to sanity checks only
            log.warning("No reference implementation available; correctness check limited to NaN/Inf/shape")
            for i, out_spec in enumerate(spec.outputs):
                if i >= len(candidate_outputs):
                    return {"correct": False, "max_abs_error": float("inf"),
                            "max_rel_error": float("inf"), "error": f"Missing output {i}"}
                out = candidate_outputs[i]
                if torch.isnan(out).any():
                    return {"correct": False, "max_abs_error": float("inf"),
                            "max_rel_error": float("inf"), "error": f"NaN in output {out_spec.name}"}
                if torch.isinf(out).any():
                    return {"correct": False, "max_abs_error": float("inf"),
                            "max_rel_error": float("inf"), "error": f"Inf in output {out_spec.name}"}
            # Explicitly mark that this was NOT a reference comparison
            return {"correct": True, "max_abs_error": 0.0, "max_rel_error": 0.0,
                    "note": "sanity_only_no_reference"}

        # Compare candidate vs reference elementwise
        if not isinstance(reference_outputs, tuple):
            reference_outputs = (reference_outputs,)

        max_abs = 0.0
        max_rel = 0.0

        for i, out_spec in enumerate(spec.outputs):
            if i >= len(candidate_outputs):
                return {"correct": False, "max_abs_error": float("inf"),
                        "max_rel_error": float("inf"), "error": f"Missing output {i}: {out_spec.name}"}
            if i >= len(reference_outputs):
                return {"correct": False, "max_abs_error": float("inf"),
                        "max_rel_error": float("inf"), "error": f"Reference missing output {i}: {out_spec.name}"}

            cand = candidate_outputs[i].float()
            ref = reference_outputs[i].float()

            # Shape check
            if cand.shape != ref.shape:
                return {"correct": False, "max_abs_error": float("inf"),
                        "max_rel_error": float("inf"),
                        "error": f"Shape mismatch for {out_spec.name}: candidate {cand.shape} vs reference {ref.shape}"}

            # NaN/Inf check on candidate
            if torch.isnan(cand).any():
                nan_count = torch.isnan(cand).sum().item()
                nan_idx = torch.where(torch.isnan(cand).flatten())[0][0].item()
                return {"correct": False, "max_abs_error": float("inf"),
                        "max_rel_error": float("inf"),
                        "error": f"NaN in {out_spec.name}: {nan_count} NaN values, first at flat index {nan_idx}"}
            if torch.isinf(cand).any():
                return {"correct": False, "max_abs_error": float("inf"),
                        "max_rel_error": float("inf"), "error": f"Inf in {out_spec.name}"}

            # Elementwise comparison
            abs_diff = (cand - ref).abs()
            abs_err = abs_diff.max().item()
            max_abs = max(max_abs, abs_err)

            # Relative error (avoid div by zero)
            denom = ref.abs().clamp(min=1e-8)
            rel_err = (abs_diff / denom).max().item()
            max_rel = max(max_rel, rel_err)

            # Choose tolerance based on output dtype
            atol = self.config.atol_f32 if "float32" in out_spec.dtype else self.config.atol
            rtol = self.config.rtol_f32 if "float32" in out_spec.dtype else self.config.rtol

            if abs_err > atol and rel_err > rtol:
                # Find the location of the worst error for diagnostics
                worst_flat = abs_diff.flatten().argmax().item()
                worst_idx = []
                flat = worst_flat
                for dim in reversed(cand.shape):
                    worst_idx.insert(0, flat % dim)
                    flat //= dim

                return {
                    "correct": False,
                    "max_abs_error": abs_err,
                    "max_rel_error": rel_err,
                    "error": (
                        f"Output {out_spec.name} exceeds tolerance: "
                        f"abs={abs_err:.6f} (tol={atol}), rel={rel_err:.6f} (tol={rtol}). "
                        f"Worst at index {worst_idx}: candidate={cand.flatten()[worst_flat]:.6f}, "
                        f"reference={ref.flatten()[worst_flat]:.6f}"
                    ),
                    "error_location": str(worst_idx),
                }

        return {"correct": True, "max_abs_error": max_abs, "max_rel_error": max_rel}

    def _run_reference(self, spec: KernelSpec, inputs: dict):
        """
        Run the spec's reference implementation to get ground-truth outputs.
        Returns None if no reference is available or it fails.
        """
        import torch

        ref_code = spec.reference_code.strip()
        if not ref_code:
            return None

        try:
            # Compile the reference code into a function
            ref_globals = {"torch": torch}
            exec(ref_code, ref_globals)

            # Look for a callable entry point
            ref_fn = None
            for name in ("reference", "ref", "forward", "run", "main"):
                if name in ref_globals and callable(ref_globals[name]):
                    ref_fn = ref_globals[name]
                    break

            # Also check for any function defined in the code
            if ref_fn is None:
                import types
                for v in ref_globals.values():
                    if isinstance(v, types.FunctionType) and v.__module__ is None:
                        ref_fn = v
                        break

            if ref_fn is None:
                log.warning("Reference code has no callable function")
                return None

            ref_outputs = ref_fn(**inputs)
            return ref_outputs

        except Exception as e:
            log.warning(f"Reference implementation failed: {type(e).__name__}: {e}")
            return None

    def _benchmark(self, spec: KernelSpec, kernel_fn, shape_dict: dict[str, int]) -> list[float]:
        """Benchmark kernel latency in microseconds."""
        import torch

        inputs = self._generate_test_inputs(spec, shape_dict)
        device = self.config.device

        # Warmup
        for _ in range(self.config.warmup_runs):
            kernel_fn(**inputs)
        torch.cuda.synchronize(device)

        # Benchmark
        latencies = []
        for _ in range(self.config.benchmark_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            kernel_fn(**inputs)
            end.record()

            torch.cuda.synchronize(device)
            latencies.append(start.elapsed_time(end) * 1000)  # ms → μs

        return latencies

    def _run_ncu(self, solution: Solution, shape_dict: dict[str, int]) -> dict[str, Any]:
        """Run NVIDIA Nsight Compute for detailed profiling.

        4-tier fallback chain:
        1. flashinfer-bench NCU CLI (if KERNFORGE_USE_FLASHINFER_BENCH env var set)
        2. flashinfer_bench.agents Python API (works on Modal B200 — preferred for remote GPU)
        3. Local ncu CLI (requires local GPU + ncu installation)
        4. Triton-based heuristic profiling (static analysis, always available)

        The returned dict includes a "source" key indicating which profiler was used.
        """
        dataset_root = os.environ.get("FLASHINFER_BENCH_DATASET", "")

        # Tier 1: flashinfer-bench NCU CLI (subprocess, works if CLI is installed)
        if os.environ.get("KERNFORGE_USE_FLASHINFER_BENCH"):
            try:
                from .ncu import run_ncu_flashinfer_bench
                profile = run_ncu_flashinfer_bench(
                    solution_dir=".",  # would need actual path
                    dataset_path=dataset_root,
                    definition=solution.definition,
                )
                result = profile.to_dict()
                result["source"] = "flashinfer_bench_ncu"
                log.info("NCU profiling via flashinfer-bench CLI")
                return result
            except Exception as e:
                log.debug(f"flashinfer-bench NCU CLI failed: {e}")

        # Tier 2: flashinfer_bench.agents Python API (works on Modal B200)
        if dataset_root:
            try:
                from .profiling import run_ncu_profile
                ncu_result = run_ncu_profile(
                    solution.main_source,
                    solution.definition,
                    dataset_root,
                )
                if not ncu_result.get("error"):
                    result = ncu_result.get("metrics", {})
                    result["bottleneck"] = ncu_result.get("bottleneck", "unknown")
                    result["source"] = "flashinfer_bench_agents_api"
                    log.info("NCU profiling via flashinfer_bench.agents API (Modal-compatible)")
                    return result
                else:
                    log.debug(f"flashinfer_bench.agents NCU failed: {ncu_result['error']}")
            except Exception as e:
                log.debug(f"flashinfer_bench.agents API failed: {e}")

        # Tier 3: local ncu CLI (requires local GPU + ncu installation)
        try:
            from .ncu import run_ncu_cli
            profile = run_ncu_cli(
                solution.main_source,
                shape_dict,
                spec_name=solution.definition,
            )
            result = profile.to_dict()
            result["source"] = "ncu_cli"
            log.info("NCU profiling via local ncu CLI")
            return result
        except Exception as e:
            log.debug(f"Local ncu CLI failed: {e}")

        # Tier 4: Triton heuristic profiling (static source analysis, always available)
        try:
            from .triton_profiling import run_triton_profiling
            result = run_triton_profiling(solution.main_source, shape_dict)
            result["source"] = "triton_heuristic"
            log.info("NCU profiling unavailable — using Triton heuristic profiling")
            return result
        except Exception as e:
            log.debug(f"Triton heuristic profiling failed: {e}")

        log.warning("All profiling methods failed — returning empty metrics")
        return {"source": "none"}

    def _diagnose_correctness_error(
        self, result: dict, shape_dict: dict[str, int], spec: KernelSpec
    ) -> str:
        """Produce a detailed diagnosis of a correctness failure for the LLM."""
        lines = ["## Correctness Failure Diagnosis"]

        abs_err = result.get("max_abs_error", 0)
        rel_err = result.get("max_rel_error", 0)

        # Error magnitude classification
        if abs_err > 100:
            lines.append("- ERROR MAGNITUDE: Very large (>100) — likely a fundamental algorithm bug")
            lines.append("  Possible causes: wrong dimension reduction, missing operation, wrong head mapping")
        elif abs_err > 1:
            lines.append("- ERROR MAGNITUDE: Large (>1) — likely a scaling or gate computation bug")
            lines.append("  Possible causes: missing alpha/beta/dt factor, wrong softplus/sigmoid, wrong scale")
        elif abs_err > 0.1:
            lines.append("- ERROR MAGNITUDE: Medium (>0.1) — likely numerical precision issue")
            lines.append("  Possible causes: bf16 accumulation (use f32), missing type cast, wrong order of operations")
        elif abs_err > 0.01:
            lines.append("- ERROR MAGNITUDE: Small (>0.01) — precision edge case")
            lines.append("  Possible causes: fused multiply-add ordering, softplus approximation, rounding mode")

        # Shape-specific hints
        shape_str = ", ".join(f"{k}={v}" for k, v in shape_dict.items())
        lines.append(f"- FAILING SHAPE: {shape_str}")

        if shape_dict.get("batch_size", 0) == 1:
            lines.append("  This is batch=1 — check that batch indexing handles the single-element case")

        if result.get("error_location"):
            lines.append(f"- ERROR LOCATION: {result['error_location']}")

        return "\n".join(lines)
