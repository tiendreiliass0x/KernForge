"""
Adversarial Test Case Generation — stress edge cases that break kernels.

LLMs write kernels that work for "nice" shapes (batch=16, heads=32, D=128)
but break on edge cases. This module generates shapes designed to expose:

1. Boundary conditions: batch=1, seq_len=1
2. Non-power-of-2 dimensions: batch=3, heads=7
3. Tile boundary misalignment: shapes that don't divide evenly by BLOCK sizes
4. Large scale: maximum supported dimensions
5. Numerical stress: inputs that trigger overflow/underflow
6. GDN-specific: single-head, GVA ratio edge cases, zero-length sequences
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case with shape and special input properties."""
    name: str
    shape: dict[str, int]
    category: str  # "boundary", "alignment", "scale", "numerical", "gdn_specific"
    description: str
    input_overrides: dict[str, Any] = field(default_factory=dict)
    # e.g., {"A_log": "large_positive"} to test sigmoid saturation


@dataclass
class TestSuite:
    """A collection of test cases for a kernel."""
    cases: list[TestCase]
    kernel_type: str

    def shapes_only(self) -> list[dict[str, int]]:
        """Just the shape dicts for quick correctness checking."""
        return [tc.shape for tc in self.cases]

    def to_prompt_context(self) -> str:
        """Format as test plan for the LLM."""
        lines = [f"## Test Suite ({len(self.cases)} cases)"]
        for tc in self.cases:
            lines.append(f"- **{tc.name}** [{tc.category}]: {tc.shape}")
            lines.append(f"  {tc.description}")
        return "\n".join(lines)


def generate_test_suite(kernel_type: str, spec_axes: list | None = None) -> TestSuite:
    """Generate a comprehensive adversarial test suite for a kernel type."""
    if "gdn" in kernel_type or "gated_delta" in kernel_type:
        return _gdn_test_suite(kernel_type)
    elif "attention" in kernel_type:
        return _attention_test_suite()
    else:
        return _generic_test_suite(spec_axes)


# ============================================================================
# GDN-specific test cases
# ============================================================================

def _gdn_test_suite(kernel_type: str) -> TestSuite:
    """Adversarial tests for Gated Delta Net kernels."""
    is_decode = "decode" in kernel_type or "prefill" not in kernel_type

    cases = [
        # === Boundary conditions ===
        TestCase(
            name="single_batch",
            shape={"batch_size": 1, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
            category="boundary",
            description="Minimum batch. Grid may underutilize SMs (1*32=32 blocks < 192 SMs on B200).",
        ),
        TestCase(
            name="single_head",
            shape={"batch_size": 4, "seq_len": 1, "num_q_heads": 1, "num_k_heads": 1, "num_v_heads": 2, "head_size": 128},
            category="boundary",
            description="Minimal head count. Tests GVA mapping with ratio=2 at minimum.",
        ),
        TestCase(
            name="gva_ratio_1",
            shape={"batch_size": 4, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 16, "head_size": 128},
            category="boundary",
            description="GVA ratio = 1 (no grouping). Tests that the GVA mapping handles identity case.",
        ),

        # === Tile alignment ===
        TestCase(
            name="non_pow2_batch",
            shape={"batch_size": 3, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
            category="alignment",
            description="Non-power-of-2 batch. Tests masking in batch dimension.",
        ),
        TestCase(
            name="v_not_divisible_by_block",
            shape={"batch_size": 4, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 48, "head_size": 128},
            category="alignment",
            description="num_v_heads=48 not divisible by common BLOCK_V sizes. Tests tile boundary masking. GVA ratio=3.",
        ),

        # === Scale ===
        TestCase(
            name="large_batch",
            shape={"batch_size": 128, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
            category="scale",
            description="Large batch. 128*32=4096 blocks — tests scheduler and memory at scale.",
        ),
        TestCase(
            name="max_heads",
            shape={"batch_size": 4, "seq_len": 1, "num_q_heads": 64, "num_k_heads": 64, "num_v_heads": 128, "head_size": 128},
            category="scale",
            description="Many heads. Tests grid sizing and memory bandwidth at high head count.",
        ),

        # === Numerical stress ===
        TestCase(
            name="numerical_large_state",
            shape={"batch_size": 4, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
            category="numerical",
            description="Normal shape but state values are large (1e3 range). Tests f32 accumulation stability.",
            input_overrides={"state_scale": 1000.0},
        ),
        TestCase(
            name="numerical_tiny_dt",
            shape={"batch_size": 4, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
            category="numerical",
            description="Very small dt (softplus near zero). Tests that the delta write doesn't vanish.",
            input_overrides={"dt_bias_value": -10.0},
        ),
        TestCase(
            name="numerical_saturated_gates",
            shape={"batch_size": 4, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
            category="numerical",
            description="A_log and b very large/small → sigmoid saturated to 0 or 1. Tests gate boundary behavior.",
            input_overrides={"A_log_value": 20.0, "b_value": -20.0},
        ),

        # === The contest benchmark shape ===
        TestCase(
            name="contest_typical",
            shape={"batch_size": 64, "seq_len": 1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
            category="benchmark",
            description="Typical contest evaluation shape. This is what gets benchmarked.",
        ),
    ]

    # Prefill-specific cases
    if not is_decode:
        cases.extend([
            TestCase(
                name="short_sequence",
                shape={"batch_size": 4, "seq_len": 8, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
                category="boundary",
                description="Sequence shorter than chunk size. Tests single-chunk handling.",
            ),
            TestCase(
                name="seq_not_divisible_by_chunk",
                shape={"batch_size": 4, "seq_len": 100, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
                category="alignment",
                description="seq_len=100 not divisible by chunk_size=64. Tests remainder chunk handling.",
            ),
            TestCase(
                name="long_sequence",
                shape={"batch_size": 2, "seq_len": 2048, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
                category="scale",
                description="Long sequence. Tests chunkwise parallelism and state accumulation over many chunks.",
            ),
            TestCase(
                name="variable_lengths",
                shape={"batch_size": 4, "seq_len": -1, "num_q_heads": 16, "num_k_heads": 16, "num_v_heads": 32, "head_size": 128},
                category="gdn_specific",
                description="Variable-length sequences via cu_seqlens: [0, 32, 96, 128, 256]. Tests packed sequence handling.",
                input_overrides={"cu_seqlens": [0, 32, 96, 128, 256]},
            ),
        ])

    return TestSuite(cases=cases, kernel_type=kernel_type)


def _attention_test_suite() -> TestSuite:
    """Adversarial tests for attention kernels."""
    return TestSuite(
        cases=[
            TestCase("batch_1", {"batch": 1, "heads": 8, "seq_len": 64, "d": 128}, "boundary", "Single batch"),
            TestCase("seq_1", {"batch": 4, "heads": 8, "seq_len": 1, "d": 128}, "boundary", "Single token"),
            TestCase("non_pow2_seq", {"batch": 4, "heads": 8, "seq_len": 100, "d": 128}, "alignment", "Non-pow2 seq"),
            TestCase("large", {"batch": 64, "heads": 32, "seq_len": 4096, "d": 128}, "scale", "Large scale"),
        ],
        kernel_type="attention",
    )


def _generic_test_suite(spec_axes: list | None) -> TestSuite:
    """Generic adversarial tests based on spec axes."""
    cases = [
        TestCase("minimal", {}, "boundary", "All dimensions at minimum"),
        TestCase("typical", {}, "benchmark", "Typical workload shape"),
        TestCase("maximum", {}, "scale", "All dimensions at maximum supported"),
    ]
    return TestSuite(cases=cases, kernel_type="generic")


# ============================================================================
# Input generation for adversarial cases
# ============================================================================

def generate_adversarial_inputs(
    test_case: TestCase,
    spec,
    device: str = "cuda:0",
) -> dict:
    """Generate inputs for an adversarial test case, including special overrides."""
    import torch

    shape_dict = test_case.shape
    overrides = test_case.input_overrides

    # Start with standard random inputs (delegate to evaluator's method)
    # Then apply overrides for adversarial conditions

    inputs = {}  # Base inputs would come from evaluator._generate_test_inputs

    # Apply numerical overrides
    if "state_scale" in overrides:
        scale = overrides["state_scale"]
        if "state" in inputs:
            inputs["state"] = inputs["state"] * scale

    if "dt_bias_value" in overrides:
        if "dt_bias" in inputs:
            inputs["dt_bias"] = torch.full_like(inputs["dt_bias"], overrides["dt_bias_value"])

    if "A_log_value" in overrides:
        if "A_log" in inputs:
            inputs["A_log"] = torch.full_like(inputs["A_log"], overrides["A_log_value"])

    if "b_value" in overrides:
        if "b" in inputs:
            inputs["b"] = torch.full_like(inputs["b"], overrides["b_value"])

    if "cu_seqlens" in overrides:
        inputs["cu_seqlens"] = torch.tensor(overrides["cu_seqlens"], dtype=torch.int32, device=device)

    return inputs
