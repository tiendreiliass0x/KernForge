#!/usr/bin/env python3
"""
Dry-run test — validates KernForge components work without GPU or API keys.
Run this to verify the installation and structure are correct.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_spec_loading():
    """Test that kernel specs can be loaded and formatted."""
    from kernforge.kernel.spec import KernelSpec

    template = Path(__file__).parent.parent / "kernforge" / "templates" / "gdn_decode_qk16_v32_d128_k_last.json"
    spec = KernelSpec.from_file(template)

    assert spec.name == "gdn_decode_qk16_v32_d128_k_last"
    assert spec.kernel_type == "gdn"
    assert len(spec.inputs) == 9
    assert len(spec.outputs) == 2
    assert spec.const_axes["num_v_heads"] == 32
    assert spec.const_axes["head_size"] == 128
    assert "batch_size" in spec.var_axes

    context = spec.to_prompt_context()
    assert "128" in context
    assert "bfloat16" in context
    print("✅ Spec loading: PASSED")
    return spec


def test_solution_creation():
    """Test that solutions can be created and serialized."""
    from kernforge.kernel.solution import Solution, EvalResult

    sol = Solution(
        name="test-solution-v1",
        definition="gdn_decode_qk16_v32_d128_k_last",
        language="triton",
        entry_point="kernel",
        sources={"kernel.py": "import triton\n# placeholder"},
        generation=0,
        strategy="initial generation",
    )

    assert sol.id  # has a hash
    assert sol.main_source == "import triton\n# placeholder"

    # Test JSON export
    j = sol.to_flashinfer_bench_json()
    assert j["name"] == "test-solution-v1"
    assert j["spec"]["language"] == "triton"
    assert len(j["sources"]) == 1

    # Test eval result
    result = EvalResult(correct=True, median_latency_us=42.5)
    assert result.passed
    assert result.status == "passed"

    result_bad = EvalResult(correct=False, compile_error="SyntaxError: invalid syntax")
    assert not result_bad.passed
    assert result_bad.status == "compile_error"

    print("✅ Solution creation: PASSED")
    return sol


def test_hardware_specs():
    """Test GPU spec formatting."""
    from kernforge.kernel.hardware import B200, H100

    assert B200.num_sms == 192
    assert B200.shared_mem_per_sm_kb == 256
    assert H100.num_sms == 132

    ctx = B200.to_prompt_context()
    assert "B200" in ctx
    assert "192" in ctx
    print("✅ Hardware specs: PASSED")


def test_prompt_formatting():
    """Test that prompts render correctly with spec data."""
    from kernforge.agent.prompts import GENERATE_FROM_REFERENCE_PROMPT, SYSTEM_PROMPT
    from kernforge.kernel.spec import KernelSpec
    from kernforge.kernel.hardware import B200

    template = Path(__file__).parent.parent / "kernforge" / "templates" / "gdn_decode_qk16_v32_d128_k_last.json"
    spec = KernelSpec.from_file(template)

    prompt = GENERATE_FROM_REFERENCE_PROMPT.format(
        kernel_spec=spec.to_prompt_context(),
        hardware_spec=B200.to_prompt_context(),
    )

    assert "gdn_decode" in prompt
    assert "128" in prompt
    assert "B200" in prompt
    assert "bfloat16" in prompt
    assert len(SYSTEM_PROMPT) > 100

    print("✅ Prompt formatting: PASSED")
    print(f"   System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"   User prompt: {len(prompt)} chars")


def test_code_extraction():
    """Test Python code extraction from LLM responses."""
    from kernforge.agent.generator import _extract_python_code, _extract_strategy

    # Normal response
    response = """Here's the kernel:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, out_ptr):
    pass
```

This kernel implements the basic structure."""

    code = _extract_python_code(response)
    assert "import triton" in code
    assert "@triton.jit" in code

    # Strategy extraction
    response2 = "<!-- STRATEGY: Use tiled matrix multiply with BLOCK_V=64 -->\n```python\ncode```"
    strategy = _extract_strategy(response2)
    assert "tiled matrix multiply" in strategy

    print("✅ Code extraction: PASSED")


def test_evolution_config():
    """Test evolution configuration."""
    from kernforge.evolve import EvolutionConfig

    config = EvolutionConfig(max_generations=5, output_dir="/tmp/test")
    assert config.max_generations == 5
    assert config.max_consecutive_failures == 5
    assert config.improvement_threshold == 0.02

    print("✅ Evolution config: PASSED")


def main():
    print("=" * 60)
    print("🔥 KernForge Dry Run Test")
    print("=" * 60)

    tests = [
        test_spec_loading,
        test_solution_creation,
        test_hardware_specs,
        test_prompt_formatting,
        test_code_extraction,
        test_evolution_config,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAILED — {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("🎉 All tests passed! KernForge is ready.")
        print("\nNext steps:")
        print("  1. export ANTHROPIC_API_KEY=sk-ant-...")
        print("  2. python examples/evolve_gdn_decode.py --generations 3")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
