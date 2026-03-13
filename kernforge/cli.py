"""
KernForge CLI — command-line interface for kernel evolution.

Usage:
    kernforge evolve --spec <definition.json> [--gpu B200] [--generations 30]
    kernforge evolve --definition gdn_decode_qk16_v32_d128_k_last --dataset ./mlsys26-contest
    kernforge generate --spec <definition.json> --output ./output
    kernforge evaluate --solution ./solution --spec <definition.json>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def cmd_evolve(args):
    """Run the evolution loop."""
    from .agent.generator import AnthropicBackend, OpenAICompatibleBackend, KernelAgent
    from .eval.evaluator import KernelEvaluator, EvalConfig
    from .evolve import EvolutionLoop, EvolutionConfig
    from .kernel.hardware import GPU_REGISTRY, B200
    from .kernel.spec import KernelSpec

    # Load kernel spec
    if args.spec:
        spec = KernelSpec.from_file(args.spec)
    elif args.definition and args.dataset:
        spec = KernelSpec.from_flashinfer_bench(args.definition, args.dataset)
    else:
        console.print("[red]Provide either --spec <file> or --definition <name> --dataset <path>[/]")
        sys.exit(1)

    # Select GPU
    gpu = GPU_REGISTRY.get(args.gpu, B200)

    # Select LLM backend
    if args.backend == "anthropic":
        backend = AnthropicBackend(model=args.model or "claude-sonnet-4-20250514")
    elif args.backend == "openai":
        backend = OpenAICompatibleBackend(
            base_url=args.base_url or "http://localhost:8000/v1",
            model=args.model or "default",
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
        )
    else:
        backend = AnthropicBackend()

    # Configure evolution
    config = EvolutionConfig(
        max_generations=args.generations,
        output_dir=args.output,
        gpu=gpu,
        eval_config=EvalConfig(device=args.device),
    )

    # Run
    loop = EvolutionLoop.create(backend=backend, config=config)
    state = loop.run(spec)

    if state.best_solution:
        console.print(f"\n[bold green]Best solution saved to {args.output}/{spec.name}/submission/[/]")
    else:
        console.print("\n[bold red]No valid solution found.[/]")
        sys.exit(1)


def cmd_generate(args):
    """Generate a single kernel (no evolution)."""
    from .agent.generator import AnthropicBackend, KernelAgent
    from .kernel.hardware import GPU_REGISTRY, B200
    from .kernel.spec import KernelSpec

    spec = KernelSpec.from_file(args.spec) if args.spec else None
    if not spec:
        console.print("[red]--spec is required[/]")
        sys.exit(1)

    gpu = GPU_REGISTRY.get(args.gpu, B200)
    backend = AnthropicBackend(model=args.model or "claude-sonnet-4-20250514")
    agent = KernelAgent(backend=backend, gpu=gpu)

    with console.status("Generating kernel..."):
        solution = agent.generate_initial(spec)

    output_dir = Path(args.output)
    solution.save(output_dir)
    console.print(f"[green]Kernel saved to {output_dir}[/]")


def cmd_spec(args):
    """Create a kernel spec from scratch or from FlashInfer-Bench."""
    from .kernel.spec import KernelSpec

    if args.definition and args.dataset:
        spec = KernelSpec.from_flashinfer_bench(args.definition, args.dataset)
        console.print(spec.to_prompt_context())
    elif args.template:
        # Load a built-in template
        template_path = Path(__file__).parent / "templates" / f"{args.template}.json"
        if template_path.exists():
            spec = KernelSpec.from_file(template_path)
            console.print(spec.to_prompt_context())
        else:
            console.print(f"[red]Template '{args.template}' not found[/]")
    else:
        console.print("[red]Provide --definition + --dataset or --template[/]")


def main():
    parser = argparse.ArgumentParser(
        prog="kernforge",
        description="🔥 KernForge — Agentic GPU Kernel Generation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # evolve
    evolve_parser = subparsers.add_parser("evolve", help="Run kernel evolution loop")
    evolve_parser.add_argument("--spec", type=str, help="Path to kernel spec JSON")
    evolve_parser.add_argument("--definition", type=str, help="FlashInfer-Bench definition name")
    evolve_parser.add_argument("--dataset", type=str, help="Path to FlashInfer-Bench dataset")
    evolve_parser.add_argument("--gpu", type=str, default="B200", help="Target GPU (B200, H100)")
    evolve_parser.add_argument("--generations", type=int, default=30, help="Max evolution generations")
    evolve_parser.add_argument("--output", type=str, default="./evolution_output", help="Output directory")
    evolve_parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    evolve_parser.add_argument("--backend", type=str, default="anthropic", help="LLM backend")
    evolve_parser.add_argument("--model", type=str, help="LLM model name")
    evolve_parser.add_argument("--base-url", type=str, help="OpenAI-compatible API base URL")
    evolve_parser.add_argument("--api-key", type=str, help="API key")
    evolve_parser.set_defaults(func=cmd_evolve)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate a single kernel")
    gen_parser.add_argument("--spec", type=str, required=True)
    gen_parser.add_argument("--gpu", type=str, default="B200")
    gen_parser.add_argument("--output", type=str, default="./generated")
    gen_parser.add_argument("--model", type=str)
    gen_parser.set_defaults(func=cmd_generate)

    # spec
    spec_parser = subparsers.add_parser("spec", help="View or create kernel specs")
    spec_parser.add_argument("--definition", type=str)
    spec_parser.add_argument("--dataset", type=str)
    spec_parser.add_argument("--template", type=str)
    spec_parser.set_defaults(func=cmd_spec)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
