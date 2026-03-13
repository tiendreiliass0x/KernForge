#!/usr/bin/env python3
"""
KernForge End-to-End Dry-Run Test

Validates the complete competition pipeline WITHOUT requiring:
- GPU hardware
- Anthropic/OpenAI API key
- flashinfer-bench installation
- HuggingFace dataset download

Tests:
1. Static analysis gate (catches bugs before GPU)
2. Proposer prompt generation (all kernel types + domain knowledge)
3. Tuner prompt generation (all error types + NCU bottleneck feedback)
4. Code extraction (parsing LLM output)
5. str_replace editing (applying tuner edits)
6. EvalResult scoring (correct ranking)
7. Pool prompt formatting (elite + recent)
8. Corpus loading (reference implementations)
9. Ladder integration (optimization phases)
10. NCU profile parsing
11. Config parsing (baseline compatible)
12. Task file parsing
13. Kernel type detection
14. Full agent loop simulation (mock LLM + mock eval)
"""

import json
import sys
import os
import traceback
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PASSED = 0
FAILED = 0
TOTAL = 0


def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        def wrapper():
            global PASSED, FAILED, TOTAL
            TOTAL += 1
            try:
                fn()
                PASSED += 1
                print(f"  ✅ {name}")
            except Exception as e:
                FAILED += 1
                print(f"  ❌ {name}: {e}")
                if os.environ.get("VERBOSE"):
                    traceback.print_exc()
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# 1. Static Analysis
# ═══════════════════════════════════════════════════════════════

@test("Static analysis catches torch inside @triton.jit")
def test_static_torch_in_jit():
    from kernforge.eval.static_analysis import analyze
    code = '''
import triton
import triton.language as tl
import torch

@triton.jit
def bad_kernel(x_ptr, n):
    idx = tl.program_id(0)
    x = tl.load(x_ptr + idx)
    y = torch.sigmoid(x)  # BUG: torch inside triton.jit
    tl.store(x_ptr + idx, y)

def run(*args, **kwargs):
    bad_kernel[(128,)](args[0], args[0].shape[0])
'''
    result = analyze(code, entry_point="run")
    assert result.valid_python, "Should be valid Python syntax"
    has_torch_warning = any("torch" in str(i.message).lower() for i in result.issues)
    assert has_torch_warning, "Should warn about torch inside triton.jit"


@test("Static analysis passes clean kernel")
def test_static_clean():
    from kernforge.eval.static_analysis import analyze
    code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def good_kernel(x_ptr, n, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(x_ptr + idx, x * 2, mask=mask)

def run(x):
    n = x.shape[0]
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    good_kernel[grid](x, n, BLOCK=1024)
    return x
'''
    result = analyze(code, entry_point="run")
    assert result.valid_python
    blocking = [i for i in result.issues if i.severity == "error"]
    assert len(blocking) == 0, f"Clean kernel should have no blocking errors, got: {blocking}"


@test("Static analysis catches syntax errors")
def test_static_syntax():
    from kernforge.eval.static_analysis import analyze
    result = analyze("def broken(:\n  pass", entry_point="run")
    assert not result.valid_python


# ═══════════════════════════════════════════════════════════════
# 2. Proposer Prompt
# ═══════════════════════════════════════════════════════════════

@test("Proposer prompt includes domain knowledge for GDN")
def test_proposer_gdn():
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt
    tp = _make_task_params()
    prompt = generate_proposer_prompt(task_params=tp, kernel_type="gdn", gpu_name="B200")
    assert "Gated Delta Net" in prompt, "Should include GDN-specific guidance"
    assert "Missing masks" in prompt, "Should include common bug warnings"
    assert "256 KB" in prompt or "256KB" in prompt, "Should include B200 SRAM info"
    assert len(prompt) > 3000, f"Enhanced prompt should be >3000 chars, got {len(prompt)}"


@test("Proposer prompt includes domain knowledge for MOE")
def test_proposer_moe():
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt
    tp = _make_task_params()
    prompt = generate_proposer_prompt(task_params=tp, kernel_type="moe", gpu_name="B200")
    assert "Mixture of Experts" in prompt


@test("Proposer prompt includes domain knowledge for GEMM")
def test_proposer_gemm():
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt
    tp = _make_task_params()
    prompt = generate_proposer_prompt(task_params=tp, kernel_type="gemm", gpu_name="B200")
    assert "GEMM" in prompt


@test("Proposer prompt includes domain knowledge for DSA")
def test_proposer_dsa():
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt
    tp = _make_task_params()
    prompt = generate_proposer_prompt(task_params=tp, kernel_type="dsa_paged", gpu_name="B200")
    assert "Paged Attention" in prompt


@test("Proposer prompt works without domain knowledge (baseline mode)")
def test_proposer_basic():
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt
    tp = _make_task_params()
    prompt = generate_proposer_prompt(task_params=tp)
    assert "Problem Statement" in prompt
    assert "test_definition" in prompt


@test("Proposer prompt with corpus context")
def test_proposer_corpus():
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt
    from kernforge.agent.corpus import get_references, references_to_prompt
    tp = _make_task_params()
    refs = get_references("gdn_decode", max_refs=1)
    assert len(refs) > 0, "Should find GDN decode references"
    corpus_ctx = references_to_prompt(refs)
    prompt = generate_proposer_prompt(task_params=tp, kernel_type="gdn", corpus_context=corpus_ctx)
    assert "Reference Implementation" in prompt


# ═══════════════════════════════════════════════════════════════
# 3. Tuner Prompt
# ═══════════════════════════════════════════════════════════════

@test("Tuner prompt: compile error guidance")
def test_tuner_compile_error():
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt
    from kernforge.eval.flashinfer_eval import EvalResult
    tp = _make_task_params()
    prompt = generate_tuner_prompt(
        ["def run(): pass"],
        [EvalResult(compiled=False, error="undefined name 'torch'")],
        tp,
    )
    assert "Compilation Error" in prompt
    assert "torch" in prompt  # error included in guidance


@test("Tuner prompt: correctness error guidance")
def test_tuner_correctness():
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt
    from kernforge.eval.flashinfer_eval import EvalResult
    tp = _make_task_params()
    prompt = generate_tuner_prompt(
        ["def run(): pass"],
        [EvalResult(compiled=True, correct=False, error="NaN output")],
        tp,
    )
    assert "Correctness Error" in prompt


@test("Tuner prompt: performance guidance when slow")
def test_tuner_slow():
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt
    from kernforge.eval.flashinfer_eval import EvalResult
    tp = _make_task_params()
    prompt = generate_tuner_prompt(
        ["def run(): pass"],
        [EvalResult(compiled=True, correct=True, speedup=0.5)],
        tp,
    )
    assert "Performance" in prompt


@test("Tuner prompt: NCU bottleneck feedback")
def test_tuner_ncu():
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt
    from kernforge.eval.flashinfer_eval import EvalResult
    from kernforge.eval.agents_integration import NCUProfile
    tp = _make_task_params()
    ncu = NCUProfile(
        bottleneck="memory_bandwidth",
        dram_throughput_pct=72.0,
        sm_throughput_pct=15.0,
    )
    prompt = generate_tuner_prompt(
        ["def run(): pass"],
        [EvalResult(compiled=True, correct=True, speedup=0.8)],
        tp,
        ncu_profile=ncu,
    )
    assert "MEMORY BOUND" in prompt
    assert "72" in prompt  # specific percentage


@test("Tuner prompt: compute bottleneck")
def test_tuner_compute_bound():
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt
    from kernforge.eval.flashinfer_eval import EvalResult
    from kernforge.eval.agents_integration import NCUProfile
    tp = _make_task_params()
    ncu = NCUProfile(bottleneck="compute", dram_throughput_pct=20, sm_throughput_pct=85)
    prompt = generate_tuner_prompt(
        ["def run(): pass"],
        [EvalResult(compiled=True, correct=True, speedup=0.9)],
        tp,
        ncu_profile=ncu,
    )
    assert "COMPUTE BOUND" in prompt


# ═══════════════════════════════════════════════════════════════
# 4. Code Extraction & Editing
# ═══════════════════════════════════════════════════════════════

@test("Extract code from markdown block")
def test_extract_code():
    from kernforge.main import extract_first_code
    output = """Here's the optimized kernel:

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, n):
    pass

def run(x):
    kernel[(1,)](x, x.shape[0])
    return x
```

This kernel uses tiling for performance."""
    code = extract_first_code(output)
    assert "import triton" in code
    assert "def run" in code
    assert "Here's the" not in code  # preamble stripped


@test("Extract edits from tagged output")
def test_extract_edits():
    from kernforge.main import extract_edits
    output = """I'll make two edits:

<reasoning_1>
Fix the block size for better occupancy
</reasoning_1>
<old_str_1>
BLOCK = 64
</old_str_1>
<new_str_1>
BLOCK = 128
</new_str_1>

<reasoning_2>
Add mask for safety
</reasoning_2>
<old_str_2>
x = tl.load(ptr + offs)
</old_str_2>
<new_str_2>
x = tl.load(ptr + offs, mask=offs < n, other=0.0)
</new_str_2>
"""
    edits = extract_edits(output)
    assert len(edits) == 2
    assert "BLOCK = 64" in edits[0][0]
    assert "BLOCK = 128" in edits[0][1]
    assert "mask=" in edits[1][1]


@test("str_replace applies edit correctly")
def test_str_replace():
    from kernforge.main import str_replace
    code = "x = tl.load(ptr)\ny = x * 2\ntl.store(ptr, y)"
    result = str_replace(code, "x = tl.load(ptr)", "x = tl.load(ptr, mask=mask, other=0.0)")
    assert "mask=mask" in result
    assert "y = x * 2" in result  # rest preserved


@test("str_replace handles non-unique strings safely")
def test_str_replace_nonunique():
    from kernforge.main import str_replace
    code = "a = 1\nb = 2\na = 1"  # 'a = 1' appears twice
    result = str_replace(code, "a = 1", "a = 99")
    assert result == code  # no replacement (non-unique)


# ═══════════════════════════════════════════════════════════════
# 5. Scoring & EvalResult
# ═══════════════════════════════════════════════════════════════

@test("Scoring: compiled > not-compiled")
def test_score_compiled():
    from kernforge.eval.flashinfer_eval import EvalResult, calculate_score
    assert calculate_score(EvalResult(compiled=True, correct=False)) > \
           calculate_score(EvalResult(compiled=False))


@test("Scoring: correct > incorrect")
def test_score_correct():
    from kernforge.eval.flashinfer_eval import EvalResult, calculate_score
    assert calculate_score(EvalResult(compiled=True, correct=True, speedup=0.5)) > \
           calculate_score(EvalResult(compiled=True, correct=False))


@test("Scoring: faster > slower (both correct)")
def test_score_speed():
    from kernforge.eval.flashinfer_eval import EvalResult, calculate_score
    assert calculate_score(EvalResult(compiled=True, correct=True, speedup=2.0)) > \
           calculate_score(EvalResult(compiled=True, correct=True, speedup=1.0))


@test("Scoring: monotonic progression")
def test_score_monotonic():
    from kernforge.eval.flashinfer_eval import EvalResult, calculate_score
    scores = [
        calculate_score(None),
        calculate_score(EvalResult()),
        calculate_score(EvalResult(compiled=True)),
        calculate_score(EvalResult(compiled=True, correct=True, speedup=0.5)),
        calculate_score(EvalResult(compiled=True, correct=True, speedup=1.0)),
        calculate_score(EvalResult(compiled=True, correct=True, speedup=2.0)),
    ]
    assert scores == sorted(scores), f"Scores not monotonic: {scores}"


# ═══════════════════════════════════════════════════════════════
# 6. Pool Prompt
# ═══════════════════════════════════════════════════════════════

@test("Pool prompt formats elite + recent correctly")
def test_pool_prompt():
    from kernforge.prompt.proposer_prompt import generate_pool_prompt
    from kernforge.eval.flashinfer_eval import EvalResult
    pool = generate_pool_prompt(
        kernel_pool=["def run(): return 1"],
        metrics_pool=[EvalResult(compiled=True, correct=True, speedup=1.0)],
        elite_kernel_pool=["def run(): return 2"],
        elite_metrics_pool=[EvalResult(compiled=True, correct=True, speedup=2.0)],
    )
    assert "elite" in pool
    assert "recent" in pool


@test("Empty pool returns empty string")
def test_pool_empty():
    from kernforge.prompt.proposer_prompt import generate_pool_prompt
    assert generate_pool_prompt(kernel_pool=[], metrics_pool=[]) == ""


# ═══════════════════════════════════════════════════════════════
# 7. Corpus
# ═══════════════════════════════════════════════════════════════

@test("Corpus has GDN decode reference")
def test_corpus_gdn():
    from kernforge.agent.corpus import get_references
    refs = get_references("gdn_decode")
    assert len(refs) > 0
    assert "gdn_decode_kernel" in refs[0].source or "GDN" in refs[0].name


@test("Corpus formats as prompt correctly")
def test_corpus_prompt():
    from kernforge.agent.corpus import get_references, references_to_prompt
    refs = get_references("gdn_decode", max_refs=1)
    prompt = references_to_prompt(refs)
    assert "Reference Implementation" in prompt
    assert "Key Design Choices" in prompt


# ═══════════════════════════════════════════════════════════════
# 8. NCU Profile Parsing
# ═══════════════════════════════════════════════════════════════

@test("NCU profile parsing: memory bound")
def test_ncu_memory():
    from kernforge.eval.agents_integration import _parse_ncu_output
    output = "DRAM Throughput: 72.3%\nSM Throughput: 15.2%\nOccupancy: 45%"
    profile = _parse_ncu_output(output)
    assert profile.bottleneck == "memory_bandwidth"
    assert profile.dram_throughput_pct > 70


@test("NCU profile parsing: compute bound")
def test_ncu_compute():
    from kernforge.eval.agents_integration import _parse_ncu_output
    output = "DRAM Throughput: 12%\nSM Throughput: 85%\nOccupancy: 50%"
    profile = _parse_ncu_output(output)
    assert profile.bottleneck == "compute"


@test("NCU profile parsing: register spills")
def test_ncu_spills():
    from kernforge.eval.agents_integration import _parse_ncu_output
    output = "Local Load Throughput: 1024\nLocal Store: 512"
    profile = _parse_ncu_output(output)
    assert profile.bottleneck == "register_spills"


# ═══════════════════════════════════════════════════════════════
# 9. Config Parsing
# ═══════════════════════════════════════════════════════════════

@test("Config files parse correctly")
def test_config_parsing():
    import yaml
    config_dir = Path(__file__).parent.parent / "config"
    for cfg_file in config_dir.glob("config_*.yaml"):
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        assert "test_source" in cfg
        assert "agent_type" in cfg


@test("Baseline configs also parse")
def test_baseline_config():
    import yaml
    baseline_dir = Path.home() / "mlsys26-agent-baseline" / "config"
    if baseline_dir.exists():
        for cfg_file in baseline_dir.glob("*.yaml"):
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            assert "test_source" in cfg


# ═══════════════════════════════════════════════════════════════
# 10. Kernel Type Detection
# ═══════════════════════════════════════════════════════════════

@test("Detect kernel types from level/problem_id")
def test_kernel_types():
    from kernforge.main import detect_kernel_type
    assert detect_kernel_type("gdn", "gdn_decode_bf16_b4_h32") == "gdn"
    assert detect_kernel_type("moe", "moe_fp8_block_scale") == "moe"
    assert detect_kernel_type("gemm", "gemm_n128_k2048") == "gemm"
    assert detect_kernel_type("dsa_paged", "dsa_paged_fp16") == "dsa_paged"
    assert detect_kernel_type("custom_op", "my_kernel") == "general"


# ═══════════════════════════════════════════════════════════════
# 11. Full Agent Simulation (Mock)
# ═══════════════════════════════════════════════════════════════

@test("Full pipeline simulation: propose → static check → eval → tune")
def test_full_pipeline_simulation():
    """Simulate the complete agent loop with mocked LLM and eval."""
    from kernforge.eval.flashinfer_eval import EvalResult, calculate_score
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt
    from kernforge.main import extract_first_code, extract_edits, str_replace
    from kernforge.eval.static_analysis import analyze

    tp = _make_task_params()

    # Step 1: Generate proposer prompt
    prompt = generate_proposer_prompt(task_params=tp, kernel_type="gdn", gpu_name="B200")
    assert len(prompt) > 2000

    # Step 2: Mock LLM response
    mock_kernel = '''import torch
import triton
import triton.language as tl

@triton.jit
def gdn_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, x * 2.0, mask=mask)

def run(x):
    out = torch.empty_like(x)
    n = x.shape[0]
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    gdn_kernel[grid](x, out, n, BLOCK=1024)
    return out
'''

    # Step 3: Static analysis
    analysis = analyze(mock_kernel, entry_point="run")
    assert analysis.valid_python
    blocking = [i for i in analysis.issues if i.severity == "error"]
    assert len(blocking) == 0, "Mock kernel should pass static analysis"

    # Step 4: Mock eval result
    mock_eval = EvalResult(compiled=True, correct=True, speedup=0.8, task_id="test")

    # Step 5: Generate tuner prompt with eval feedback
    tuner = generate_tuner_prompt(
        previous_kernels=[mock_kernel],
        previous_metrics=[mock_eval],
        task_params=tp,
    )
    assert "str_replace" in tuner
    assert "0.8" in tuner or "Performance" in tuner

    # Step 6: Mock tuner response (edit)
    mock_edit_response = """<reasoning_1>
The kernel should use a larger block size for better GPU utilization.
</reasoning_1>
<old_str_1>
    gdn_kernel[grid](x, out, n, BLOCK=1024)
</old_str_1>
<new_str_1>
    gdn_kernel[grid](x, out, n, BLOCK=2048)
</new_str_1>
"""
    edits = extract_edits(mock_edit_response)
    assert len(edits) == 1

    # Step 7: Apply edit
    tuned_kernel = mock_kernel
    for old, new in edits:
        tuned_kernel = str_replace(tuned_kernel, old, new)
    assert "BLOCK=2048" in tuned_kernel

    # Step 8: Re-analyze
    analysis2 = analyze(tuned_kernel, entry_point="run")
    assert analysis2.valid_python

    # Step 9: Scoring
    mock_eval2 = EvalResult(compiled=True, correct=True, speedup=1.2, task_id="test")
    assert calculate_score(mock_eval2) > calculate_score(mock_eval)


# ═══════════════════════════════════════════════════════════════
# 12. Submission Packing
# ═══════════════════════════════════════════════════════════════

@test("Manual submission packing")
def test_submission_pack():
    import tempfile
    from kernforge.eval.agents_integration import _manual_pack
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _manual_pack(
            "def run(): pass",
            "test_definition",
            "gated_delta_net",
            "test_author",
            tmpdir,
        )
        assert os.path.exists(path)
        with open(path) as f:
            solution = json.load(f)
        assert solution["definition"] == "test_definition"
        assert solution["spec"]["entry_point"] == "main.py::run"
        assert solution["sources"][0]["content"] == "def run(): pass"


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_task_params():
    return {
        "definition": json.dumps({"name": "test_definition", "inputs": []}),
        "target_gpu": "B200",
        "gpu_name": "B200",
        "gpu_architecture": "Blackwell",
        "dtype_str": "bfloat16",
    }


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("KernForge End-to-End Dry-Run Test")
    print("=" * 60)
    print()

    # Collect all test functions
    tests = []
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            tests.append(obj)

    # Run all tests
    for t in tests:
        t()

    print()
    print("=" * 60)
    if FAILED == 0:
        print(f"ALL {PASSED}/{TOTAL} TESTS PASSED ✅")
    else:
        print(f"FAILED: {FAILED}/{TOTAL} tests")
        print(f"PASSED: {PASSED}/{TOTAL}")
    print("=" * 60)

    sys.exit(1 if FAILED > 0 else 0)
