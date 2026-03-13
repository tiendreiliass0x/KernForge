"""
Kernel evaluation using flashinfer-bench Python API directly.
Compatible with the official baseline's eval interface, plus KernForge enhancements.
"""

import json
import logging
import uuid
from typing import Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    """Result of evaluating a single kernel. Matches baseline format exactly."""
    compiled: bool = False
    correct: bool = False
    speedup: float = 0.0
    latency_ms: float | None = None
    task_id: str = ""
    error: str | None = None
    stats: dict | None = None


def calculate_score(metric: EvalResult | None):
    """Return (compiled, correct, speedup) tuple for ranking. Matches baseline."""
    if metric is None:
        return (0, 0, 0)
    if not metric.compiled:
        return (0, 0, 0)
    if not metric.correct:
        return (1, 0, 0)
    return (1, 1, metric.speedup)


def read_metrics(metrics_path: str, full: bool = False):
    """Read metrics from a JSON file."""
    with open(metrics_path, "r") as f:
        data = json.load(f)
    if full:
        return EvalResult(**data)
    if data.get("compiled") and data.get("correct"):
        return (True, data.get("speedup", 0.0))
    return (False, 0.0)


def eval_kernel(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
    backend: str = "triton",
    timeout: int = 60,
) -> EvalResult:
    """
    Evaluate a kernel against the reference using flashinfer-bench API.
    This is the REAL eval, not a stub. Uses the same API as the official baseline.
    """
    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import (
        BuildSpec,
        EvaluationStatus,
        Solution,
        SourceFile,
        SupportedLanguages,
        TraceSet,
    )

    trace_set = TraceSet.from_path(dataset_root)

    solution_name = f"kernforge_{uuid.uuid4().hex[:8]}"
    language = (
        SupportedLanguages.TRITON if backend == "triton" else SupportedLanguages.CUDA
    )

    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="kernforge",
        spec=BuildSpec(
            language=language,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            dependencies=[],
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=kernel_code)],
    )

    # Inject solution into in-memory trace set
    trace_set.solutions.setdefault(task_id, []).append(solution)
    trace_set._solution_by_name[solution_name] = solution

    config = BenchmarkConfig(
        warmup_runs=3,
        iterations=5,
        num_trials=1,
        definitions=[task_id],
        solutions=[solution_name],
        timeout_seconds=timeout,
    )

    benchmark = Benchmark(trace_set, config)
    try:
        result_ts = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()

    traces = result_ts.traces.get(task_id, [])

    # Check for errors
    error_statuses = {
        EvaluationStatus.COMPILE_ERROR,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_NUMERICAL,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.TIMEOUT,
    }
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status in error_statuses:
            return EvalResult(
                compiled=(ev.status != EvaluationStatus.COMPILE_ERROR),
                task_id=task_id,
                error=f"{ev.status.value}: {ev.log}",
            )

    # Aggregate PASSED results
    latencies, ref_latencies, speedups = [], [], []
    rel_errors, abs_errors = [], []
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status == EvaluationStatus.PASSED:
            latencies.append(ev.performance.latency_ms)
            ref_latencies.append(ev.performance.reference_latency_ms)
            speedups.append(ev.performance.speedup_factor)
            rel_errors.append(ev.correctness.max_relative_error)
            abs_errors.append(ev.correctness.max_absolute_error)

    if not latencies:
        return EvalResult(task_id=task_id, error="No evaluation results")

    n = len(latencies)
    return EvalResult(
        compiled=True,
        correct=True,
        speedup=sum(speedups) / n,
        latency_ms=sum(latencies) / n,
        task_id=task_id,
        stats={
            "reference_latency_ms": sum(ref_latencies) / n,
            "max_relative_error": sum(rel_errors) / n,
            "max_absolute_error": sum(abs_errors) / n,
            "total_workloads": n,
        },
    )


def eval_kernel_with_static_check(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
    backend: str = "triton",
    timeout: int = 60,
) -> EvalResult:
    """
    KernForge enhanced eval: run static analysis BEFORE GPU eval.
    Catches ~60% of bugs for free (no GPU time, no API cost).
    Falls through to real eval only if static analysis passes.
    """
    from kernforge.eval.static_analysis import analyze

    # Static analysis gate
    analysis = analyze(kernel_code, entry_point="run")
    if analysis.blocking:
        error_msgs = [
            f"L{issue.line}: {issue.message}" for issue in analysis.issues
            if issue.severity == "error"
        ]
        suggestions = [
            issue.suggestion for issue in analysis.issues
            if issue.suggestion and issue.severity == "error"
        ]
        return EvalResult(
            compiled=False,
            task_id=task_id,
            error=(
                f"Static analysis found {len(error_msgs)} blocking error(s) "
                f"(caught before GPU eval):\n"
                + "\n".join(error_msgs[:5])
                + ("\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions[:3])
                   if suggestions else "")
            ),
        )

    # Warnings go to logs but don't block
    warnings = [i for i in analysis.issues if i.severity == "warning"]
    if warnings:
        logger.info(
            f"Static analysis: {len(warnings)} warning(s) for {task_id} "
            f"(proceeding to GPU eval)"
        )

    # Pass to real eval
    return eval_kernel(kernel_code, task_id, dataset_root, backend, timeout)


def create_eval_fn(
    backend: str = "local",
    dataset_name: str = "mlsys26-contest",
    remote_fn=None,
    use_static_check: bool = True,
) -> Callable:
    """
    Factory to create eval function. Compatible with baseline interface.

    Args:
        backend: "local" for local GPU, "modal" for Modal remote GPU.
        dataset_name: Dataset subdirectory name (for modal).
        remote_fn: Modal remote function (required for modal backend).
        use_static_check: If True, run static analysis before GPU eval (KernForge enhancement).
    """
    if backend == "local":
        if use_static_check:
            return eval_kernel_with_static_check
        return eval_kernel
    elif backend == "modal":
        if remote_fn is None:
            raise ValueError("remote_fn is required for modal backend")

        base_eval = _make_modal_eval(remote_fn, dataset_name)
        if not use_static_check:
            return base_eval

        # Wrap modal eval with static check
        def _modal_with_static(kernel_code, task_id, dataset_root, backend="triton", timeout=60):
            from kernforge.eval.static_analysis import analyze
            analysis = analyze(kernel_code, entry_point="run")
            if analysis.blocking:
                error_msgs = [f"L{i.line}: {i.message}" for i in analysis.issues if i.severity == "error"]
                return EvalResult(
                    compiled=False, task_id=task_id,
                    error=f"Static analysis: {'; '.join(error_msgs[:3])}",
                )
            return base_eval(kernel_code, task_id, dataset_root, backend, timeout)

        return _modal_with_static
    else:
        raise ValueError(f"Unknown eval backend: {backend}")


class ModalAppStopped(Exception):
    """Raised when Modal app is stopped and needs restart."""
    pass


def _make_modal_eval(remote_fn, dataset_name: str):
    """Create a modal eval function that wraps the remote function."""
    def _modal_eval(kernel_code, task_id, dataset_root, backend="triton", timeout=60):
        try:
            result_dict = remote_fn.remote(kernel_code, task_id, dataset_name, backend, timeout)
            return EvalResult(**result_dict)
        except Exception as e:
            error_msg = str(e)
            # Detect app-stopped: raise to abort the run early instead of wasting API calls
            if "stopped" in error_msg.lower() or "disabled" in error_msg.lower():
                raise ModalAppStopped(
                    f"Modal app stopped — remaining steps cannot be evaluated. {error_msg[:200]}"
                )
            if "timeout" in error_msg.lower():
                error_msg = f"Modal eval timed out (kernel too slow or hung): {error_msg[:200]}"
            return EvalResult(compiled=True, correct=False, task_id=task_id, error=error_msg[:500])
    return _modal_eval
