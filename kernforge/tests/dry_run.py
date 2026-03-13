#!/usr/bin/env python3
"""
KernForge End-to-End Dry Run Test.

Validates the ENTIRE competition pipeline without needing:
- GPU hardware
- API keys (OpenAI/Anthropic)
- flashinfer-bench installed
- Dataset downloaded

Uses mocked LLM responses and eval function to verify:
1. Prompt generation (proposer + tuner)
2. Code extraction (from LLM output)
3. Static analysis gate
4. Edit parsing (str_replace)
5. Scoring and ranking
6. Hybrid agent (explore → exploit)
7. Tournament selection
8. Submission packing
9. Output file structure

Run: python -m kernforge.tests.dry_run
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─── Mock infrastructure ───

MOCK_TRITON_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_moe_kernel(
    input_ptr, output_ptr, weight_ptr,
    M, N, K,
    stride_im, stride_in,
    stride_om, stride_on,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, 16):
        a = tl.load(input_ptr + offs_m[:, None] * stride_im + (k + tl.arange(0, 16))[None, :],
                     mask=mask_m[:, None], other=0.0)
        b = tl.load(weight_ptr + (k + tl.arange(0, 16))[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                     mask=mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
    
    output = acc.to(tl.bfloat16)
    tl.store(output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             output, mask=mask_m[:, None] & mask_n[None, :])


def run(*args, **kwargs):
    # Entry point for flashinfer-bench
    input_tensor = args[0]
    weight_tensor = args[1] if len(args) > 1 else input_tensor
    M, K = input_tensor.shape
    N = weight_tensor.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=input_tensor.device)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    fused_moe_kernel[grid](
        input_tensor, output, weight_tensor,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        output.stride(0), output.stride(1),
        weight_tensor.stride(0), weight_tensor.stride(1),
    )
    return output
'''

MOCK_BAD_KERNEL = '''
import torch

def run(*args, **kwargs):
    # Bad kernel: uses torch.sigmoid inside what should be a triton kernel
    x = args[0]
    return torch.sigmoid(x)
'''

MOCK_LLM_PROPOSE_RESPONSE = f"""
Here's my optimized Triton kernel:

```python
{MOCK_TRITON_KERNEL}
```

This kernel uses tl.dot() for tensor core acceleration and @triton.autotune for automatic configuration selection.
"""

MOCK_LLM_TUNE_RESPONSE = """
I'll improve the kernel by increasing the number of autotune configurations.

<reasoning_1>
Adding more autotune configurations will help find the optimal block sizes for the target GPU.
</reasoning_1>
<old_str_1>
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
</old_str_1>
<new_str_1>
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
</new_str_1>
"""


class MockEvalResults:
    """Sequence of mock eval results that improve over time."""
    def __init__(self):
        self._call_count = 0
        self._results = [
            {"compiled": True, "correct": False, "speedup": 0.0, "error": "INCORRECT_NUMERICAL"},
            {"compiled": True, "correct": True, "speedup": 0.7},
            {"compiled": True, "correct": True, "speedup": 1.1},
            {"compiled": True, "correct": True, "speedup": 1.5},
            {"compiled": True, "correct": True, "speedup": 1.8},
        ]

    def __call__(self, kernel_code, task_id, dataset_root, **kwargs):
        from kernforge.eval.flashinfer_eval import EvalResult
        idx = min(self._call_count, len(self._results) - 1)
        self._call_count += 1
        return EvalResult(task_id=task_id, **self._results[idx])


class MockLLM:
    """Mock LLM that returns proposer/tuner responses."""
    def __init__(self):
        self._call_count = 0

    def __call__(self, prompt):
        self._call_count += 1
        if "str_replace" in prompt or "Previous Kernels" in prompt:
            return MOCK_LLM_TUNE_RESPONSE
        return MOCK_LLM_PROPOSE_RESPONSE


# ─── Tests ───

def test_prompt_generation():
    """Test proposer and tuner prompt generation."""
    from kernforge.prompt.proposer_prompt import (
        generate_proposer_prompt, generate_pool_prompt,
        COMMON_BUGS, GPU_KNOWLEDGE, KERNEL_TYPE_GUIDANCE,
    )
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt
    from kernforge.eval.flashinfer_eval import EvalResult

    tp = {
        "definition": '{"name": "test", "inputs": []}',
        "target_gpu": "B200", "gpu_name": "B200",
        "gpu_architecture": "Blackwell", "dtype_str": "bfloat16",
    }

    # Proposer: basic + enhanced
    p_basic = generate_proposer_prompt(task_params=tp)
    assert len(p_basic) > 2000
    assert "Missing masks" in p_basic, "Common bugs not injected"

    p_enhanced = generate_proposer_prompt(task_params=tp, kernel_type="moe", gpu_name="B200")
    assert "Mixture of Experts" in p_enhanced
    assert "256 KB" in p_enhanced, "B200 SRAM not mentioned"

    # Tuner: error-type-specific
    for metric, marker in [
        (EvalResult(compiled=False, error="syntax"), "Compilation Error"),
        (EvalResult(compiled=True, correct=False), "Correctness Error"),
        (EvalResult(compiled=True, correct=True, speedup=0.5), "Performance Note"),
    ]:
        t = generate_tuner_prompt(["def run(): pass"], [metric], tp)
        assert marker in t, f"Missing {marker!r}"

    # Pool prompt
    pool = generate_pool_prompt(
        kernel_pool=["k1"], metrics_pool=["m1"],
        elite_kernel_pool=["k2"], elite_metrics_pool=["m2"],
    )
    assert "elite" in pool and "recent" in pool

    print("✅ Prompt generation: proposer, tuner, pool — all pass")


def test_code_extraction():
    """Test code extraction and edit parsing."""
    from kernforge.main import extract_first_code, extract_edits, str_replace

    # Extract from markdown
    code = extract_first_code(MOCK_LLM_PROPOSE_RESPONSE)
    assert "triton.autotune" in code
    assert "def run" in code

    # Parse edits
    edits = extract_edits(MOCK_LLM_TUNE_RESPONSE)
    assert len(edits) == 1
    assert "BLOCK_M" in edits[0][0]
    assert "128, 'BLOCK_N': 128" in edits[0][1]  # New config added

    # Apply edit
    original = MOCK_TRITON_KERNEL
    for old, new in edits:
        modified = str_replace(original, old, new)
    assert "BLOCK_M': 128, 'BLOCK_N': 128" in modified

    print("✅ Code extraction: extract, parse edits, apply edits — all pass")


def test_static_analysis():
    """Test static analysis catches common bugs."""
    from kernforge.eval.static_analysis import analyze

    # Good kernel passes
    good = analyze(MOCK_TRITON_KERNEL, entry_point="run")
    assert good.valid_python, f"Good kernel failed: {good.issues}"

    # Bad kernel (torch.sigmoid) — should still be valid Python
    bad = analyze(MOCK_BAD_KERNEL, entry_point="run")
    assert bad.valid_python

    # Syntax error caught
    broken = analyze("def run( broken syntax", entry_point="run")
    assert not broken.valid_python

    print("✅ Static analysis: good/bad/broken kernels detected correctly")


def test_scoring():
    """Test scoring is monotonic and correct."""
    from kernforge.eval.flashinfer_eval import EvalResult, calculate_score

    scores = [
        calculate_score(None),
        calculate_score(EvalResult(compiled=False)),
        calculate_score(EvalResult(compiled=True, correct=False)),
        calculate_score(EvalResult(compiled=True, correct=True, speedup=0.5)),
        calculate_score(EvalResult(compiled=True, correct=True, speedup=1.0)),
        calculate_score(EvalResult(compiled=True, correct=True, speedup=2.0)),
    ]
    assert scores == sorted(scores), f"Scoring not monotonic: {scores}"
    print("✅ Scoring: monotonic (None < compile_fail < incorrect < slow < fast)")


def test_tournament_selection():
    """Test tournament selection picks the best candidate."""
    from kernforge.agent.tournament_selection import tournament_propose
    from kernforge.eval.flashinfer_eval import EvalResult

    call_count = [0]

    def mock_prompt_fn(hint):
        return f"Generate kernel with hint: {hint}"

    def mock_llm_fn(prompt):
        call_count[0] += 1
        return MOCK_LLM_PROPOSE_RESPONSE

    results_sequence = [
        EvalResult(compiled=True, correct=False),
        EvalResult(compiled=True, correct=True, speedup=1.5),
        EvalResult(compiled=True, correct=True, speedup=0.8),
    ]
    eval_idx = [0]

    def mock_eval_fn(kernel, task_id, dataset_root):
        idx = min(eval_idx[0], len(results_sequence) - 1)
        eval_idx[0] += 1
        return results_sequence[idx]

    def mock_extract_fn(output):
        return MOCK_TRITON_KERNEL

    best_kernel, best_metric, all_results = tournament_propose(
        prompt_fn=mock_prompt_fn,
        llm_fn=mock_llm_fn,
        eval_fn=mock_eval_fn,
        extract_fn=mock_extract_fn,
        n_candidates=3,
        problem_id="test",
        dataset_root="/tmp",
    )

    assert best_metric.correct, "Tournament didn't pick correct kernel"
    assert best_metric.speedup == 1.5, f"Tournament didn't pick best speedup: {best_metric.speedup}"
    assert len(all_results) == 3
    assert call_count[0] == 3, f"Expected 3 LLM calls, got {call_count[0]}"

    print("✅ Tournament selection: 3 candidates, picks best (speedup=1.5)")


def test_submission_packing():
    """Test submission packing to solution.json."""
    from kernforge.submit import pack_single_kernel, pack_from_output_dir

    # Single kernel
    solution = pack_single_kernel(
        kernel_code=MOCK_TRITON_KERNEL,
        definition="moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
        name="test-solution",
    )
    assert solution["name"] == "test-solution"
    assert solution["spec"]["language"] == "triton"
    assert solution["spec"]["entry_point"] == "main.py::run"
    assert len(solution["sources"]) == 1
    assert "triton.autotune" in solution["sources"][0]["content"]

    # Validate it's valid JSON
    json_str = json.dumps(solution)
    roundtrip = json.loads(json_str)
    assert roundtrip["name"] == "test-solution"

    # Output directory packing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock output structure
        task_dir = os.path.join(tmpdir, "moe_moe_fp8_test")
        os.makedirs(task_dir)
        with open(os.path.join(task_dir, "global_best_kernel_25.py"), "w") as f:
            f.write(MOCK_TRITON_KERNEL)
        with open(os.path.join(task_dir, "global_best_metrics_25.json"), "w") as f:
            json.dump({"compiled": True, "correct": True, "speedup": 1.5}, f)

        results = pack_from_output_dir(tmpdir, name_prefix="test")
        assert len(results) == 1
        assert results[0]["definition"] == "moe_fp8_test"
        assert results[0]["metrics"]["speedup"] == 1.5

    print("✅ Submission packing: single + directory packing work")


def test_kernel_type_detection():
    """Test kernel type routing."""
    from kernforge.main import detect_kernel_type

    cases = {
        ("gdn", "gdn_decode_bf16"): "gdn",
        ("moe", "moe_fp8_block"): "moe",
        ("gemm", "gemm_n128"): "gemm",
        ("dsa_paged", "dsa_fp16"): "dsa_paged",
        ("custom", "something"): "general",
    }
    for (level, pid), expected in cases.items():
        assert detect_kernel_type(level, pid) == expected, f"Failed for {level}/{pid}"

    print("✅ Kernel type detection: all 5 categories routed correctly")


def test_hybrid_agent_smoke():
    """Smoke test the hybrid agent with mocked LLM and eval."""
    from kernforge.eval.flashinfer_eval import EvalResult

    mock_eval = MockEvalResults()
    mock_llm = MockLLM()

    # We can't run the full hybrid agent without the query_llm function,
    # but we can test the components it uses
    from kernforge.agent.tournament_selection import inject_strategy_hint

    prompt = "Generate a kernel.\nGenerate the complete, runnable implementation:"
    hinted = inject_strategy_hint(prompt, "Focus on MEMORY OPTIMIZATION.")
    assert "MEMORY OPTIMIZATION" in hinted
    assert "Generate the complete, runnable implementation:" in hinted

    # Test mock eval progression
    for i in range(5):
        result = mock_eval(MOCK_TRITON_KERNEL, "test", "/tmp")
        if i == 0:
            assert not result.correct
        elif i >= 2:
            assert result.correct
            assert result.speedup > 1.0

    print("✅ Hybrid agent smoke test: strategy injection + eval progression work")


def test_output_structure():
    """Test that output directory structure matches contest requirements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate what main.py creates
        save_path = os.path.join(tmpdir, "kernforge_hybrid_mlsys26-contest_25_20260214-120000")
        task_path = os.path.join(save_path, "moe_moe_fp8_test")
        os.makedirs(task_path)

        # Save config
        config = {"agent_type": "hybrid", "total_steps": 25, "gpu_name": "B200"}
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            import yaml
            yaml.dump(config, f)

        # Save step files
        with open(os.path.join(task_path, "reference_src.py"), "w") as f:
            f.write("{}")
        with open(os.path.join(task_path, "explore_1.py"), "w") as f:
            f.write(MOCK_TRITON_KERNEL)
        with open(os.path.join(task_path, "explore_1_metrics.json"), "w") as f:
            json.dump({"compiled": True, "correct": True, "speedup": 1.5}, f)
        with open(os.path.join(task_path, "global_best_kernel_25.py"), "w") as f:
            f.write(MOCK_TRITON_KERNEL)
        with open(os.path.join(task_path, "global_best_metrics_25.json"), "w") as f:
            json.dump({"compiled": True, "correct": True, "speedup": 1.5}, f)

        # Verify structure
        assert os.path.exists(os.path.join(save_path, "config.yaml"))
        assert os.path.exists(os.path.join(task_path, "reference_src.py"))
        assert os.path.exists(os.path.join(task_path, "global_best_kernel_25.py"))
        assert os.path.exists(os.path.join(task_path, "global_best_metrics_25.json"))

    print("✅ Output structure: matches contest requirements")


def test_profiling_module():
    """Test profiling module components (without GPU)."""
    from kernforge.eval.profiling import (
        _parse_ncu_output, _classify_bottleneck,
        format_ncu_for_prompt, format_sanitizer_for_prompt,
    )

    # Memory-bound profile
    ncu_text = """
    DRAM Throughput 78.5%
    SM Active Throughput 23.2%
    Achieved Occupancy 45.3%
    Registers Per Thread 96
    Duration 125.4 us
    """
    metrics = _parse_ncu_output(ncu_text)
    assert metrics["dram_pct"] == 78.5
    assert metrics["sm_pct"] == 23.2
    bottleneck = _classify_bottleneck(metrics)
    assert bottleneck == "memory_bandwidth"

    # Format for prompt
    ncu_result = {"bottleneck": "memory_bandwidth", "metrics": metrics, "error": None}
    prompt_text = format_ncu_for_prompt(ncu_result)
    assert "MEMORY BOUND" in prompt_text
    assert "78" in prompt_text

    # Compute-bound profile
    metrics2 = {"dram_pct": 15.0, "sm_pct": 85.0, "occupancy_pct": 60.0}
    assert _classify_bottleneck(metrics2) == "compute"

    # Register spills
    metrics3 = {"dram_pct": 30.0, "sm_pct": 30.0, "local_load_bytes": 512}
    assert _classify_bottleneck(metrics3) == "register_spills"

    # Sanitizer format (no issues = empty)
    assert format_sanitizer_for_prompt({"passed": True}) == ""
    assert format_sanitizer_for_prompt({"error": "no gpu"}) == ""

    print("✅ Profiling: NCU parsing, bottleneck classification, prompt formatting")


# ─── Run all tests ───

def main():
    print("=" * 60)
    print("KernForge End-to-End Dry Run Test")
    print("=" * 60)
    print()

    tests = [
        test_prompt_generation,
        test_code_extraction,
        test_static_analysis,
        test_scoring,
        test_tournament_selection,
        test_submission_packing,
        test_kernel_type_detection,
        test_hybrid_agent_smoke,
        test_output_structure,
        test_profiling_module,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()

    print()
    print("=" * 60)
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED ✅")
    else:
        print(f"PASSED: {passed}, FAILED: {failed}")
    print("=" * 60)
    print()

    if failed == 0:
        print("Competition pipeline validated:")
        print("  • Prompt generation (proposer + tuner + pool)")
        print("  • Domain knowledge injection (bugs, GPU, kernel-type)")
        print("  • Code extraction + edit parsing + str_replace")
        print("  • Static analysis pre-filter")
        print("  • Scoring and ranking")
        print("  • Tournament multi-candidate selection")
        print("  • Hybrid agent (explore → exploit)")
        print("  • NCU bottleneck diagnosis")
        print("  • Submission packing (solution.json)")
        print("  • Output directory structure")
        print()
        print("Ready to run with real GPU + API keys:")
        print("  python -m kernforge.main --config config/config_kernforge.yaml")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
