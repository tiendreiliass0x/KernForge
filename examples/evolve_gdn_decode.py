#!/usr/bin/env python3
"""
KernForge — Quick Start Example

Evolve a Gated Delta Net decode kernel for the FlashInfer competition.

Usage:
    # With Anthropic API key
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/evolve_gdn_decode.py

    # With a local model (e.g., ollama, vllm)
    python examples/evolve_gdn_decode.py --backend openai --base-url http://localhost:8000/v1 --model qwen2.5-coder-32b

    # Quick test (3 generations)
    python examples/evolve_gdn_decode.py --generations 3
"""
import argparse
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernforge.agent.generator import AnthropicBackend, OpenAICompatibleBackend
from kernforge.eval.evaluator import EvalConfig
from kernforge.evolve import EvolutionConfig, EvolutionLoop
from kernforge.kernel.hardware import B200, H100
from kernforge.kernel.spec import KernelSpec


def main():
    parser = argparse.ArgumentParser(description="Evolve GDN decode kernel")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--backend", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--gpu", default="B200", choices=["B200", "H100"])
    parser.add_argument("--output", default="./evolution_output")
    parser.add_argument("--spec", type=str, help="Custom kernel spec JSON")
    args = parser.parse_args()

    # Load kernel spec
    if args.spec:
        spec = KernelSpec.from_file(args.spec)
    else:
        # Use built-in GDN decode template
        template_path = Path(__file__).parent.parent / "kernforge" / "templates" / "gdn_decode_qk16_v32_d128_k_last.json"
        spec = KernelSpec.from_file(template_path)

    # Create backend
    if args.backend == "anthropic":
        backend = AnthropicBackend(model=args.model or "claude-sonnet-4-20250514")
    else:
        backend = OpenAICompatibleBackend(
            base_url=args.base_url or "http://localhost:8000/v1",
            model=args.model or "default",
        )

    # Configure
    gpu = B200 if args.gpu == "B200" else H100
    config = EvolutionConfig(
        max_generations=args.generations,
        output_dir=args.output,
        gpu=gpu,
        max_consecutive_failures=5,
        analyze_every_n=3,
    )

    # Run evolution
    loop = EvolutionLoop.create(backend=backend, config=config)
    state = loop.run(spec)

    # Summary
    if state.best_solution:
        print(f"\n{'='*60}")
        print(f"Best kernel: {state.best_solution.name}")
        print(f"Latency: {state.best_latency_us:.1f} μs")
        print(f"Generations: {state.generation + 1}")
        print(f"Output: {args.output}/{spec.name}/submission/")
        print(f"{'='*60}")
    else:
        print("\nNo valid solution found. Check logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
