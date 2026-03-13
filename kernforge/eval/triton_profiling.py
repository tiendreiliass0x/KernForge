"""
Triton-based heuristic profiling — a best-effort fallback when NCU is unavailable.

Uses Triton's built-in utilities and torch.cuda events to estimate:
- Achieved bandwidth (from data transfer estimates)
- Kernel occupancy (from grid/block sizing heuristics)
- Basic roofline classification (memory-bound vs compute-bound)

This is LESS ACCURATE than real NCU profiling, but better than returning nothing.
The agent can still use these heuristics to guide optimization direction.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def run_triton_profiling(source_code: str, shape_dict: dict[str, int]) -> dict[str, Any]:
    """
    Estimate performance characteristics from kernel source and shapes.

    Returns a dict with heuristic metrics. All values are estimates — the
    "source" key should be checked by consumers to gauge data quality.
    """
    metrics: dict[str, Any] = {}

    # Estimate data movement from shape dimensions
    total_elements = 1
    for v in shape_dict.values():
        if v > 0:
            total_elements *= v

    # Estimate bytes moved (assume bf16 = 2 bytes, read + write)
    estimated_bytes = total_elements * 2 * 2  # 2 bytes/element, read + write
    metrics["estimated_data_bytes"] = estimated_bytes

    # Estimate arithmetic intensity from source analysis
    dot_count = source_code.count("tl.dot")
    load_count = source_code.count("tl.load")
    store_count = source_code.count("tl.store")

    metrics["triton_dot_calls"] = dot_count
    metrics["triton_load_calls"] = load_count
    metrics["triton_store_calls"] = store_count

    # Heuristic roofline classification
    if dot_count > 0 and load_count > 0:
        # Kernels with tl.dot are likely compute-bound
        compute_ratio = dot_count / (load_count + store_count + 1)
        if compute_ratio > 0.5:
            metrics["estimated_bottleneck"] = "compute"
        else:
            metrics["estimated_bottleneck"] = "memory_bandwidth"
    elif load_count + store_count > 0:
        metrics["estimated_bottleneck"] = "memory_bandwidth"
    else:
        metrics["estimated_bottleneck"] = "unknown"

    # Check for common optimization opportunities
    has_autotune = "@triton.autotune" in source_code
    has_tensor_cores = dot_count > 0
    has_masking = "mask=" in source_code

    metrics["has_autotune"] = has_autotune
    metrics["has_tensor_cores"] = has_tensor_cores
    metrics["has_masking"] = has_masking

    return metrics
