"""
KernForge Evolution Loop — the core orchestrator.

Takes a kernel specification and iteratively generates, evaluates,
and improves kernels until a performance target is met or a budget is exhausted.

This is the product. Everything else is plumbing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .agent.generator import KernelAgent, LLMBackend, AnthropicBackend
from .eval.evaluator import KernelEvaluator, EvalConfig
from .kernel.hardware import B200, GPUSpec
from .kernel.solution import EvalResult, Solution
from .kernel.spec import KernelSpec

log = logging.getLogger(__name__)
console = Console()


@dataclass
class EvolutionConfig:
    """Configuration for the evolution loop."""
    max_generations: int = 30
    max_consecutive_failures: int = 5
    max_fix_attempts: int = 3           # max retries for compile/runtime errors
    target_latency_us: float | None = None  # stop if we hit this
    improvement_threshold: float = 0.02  # 2% improvement required to accept
    save_all_solutions: bool = True
    output_dir: str = "./evolution_output"
    gpu: GPUSpec = field(default_factory=lambda: B200)
    eval_config: EvalConfig = field(default_factory=EvalConfig)

    # Agent config
    analyze_every_n: int = 3  # run performance analysis every N generations

    # Tournament mode (#4)
    tournament_size: int = 1      # 1 = disabled, 3-5 = tournament mode
    tournament_parallel: bool = True  # parallelize candidate generation

    # Strategy learning (#5)
    enable_learning: bool = True  # persist strategy outcomes
    strategy_db_path: str | None = None  # path to strategy database

    # Reference corpus (#3)
    use_corpus: bool = True  # inject reference kernels as few-shot examples
    fetch_fla_refs: bool = False  # fetch FLA's kernels from GitHub


@dataclass
class EvolutionState:
    """Tracks the state of an evolution run."""
    spec: KernelSpec
    best_solution: Solution | None = None
    best_latency_us: float = float("inf")
    current_solution: Solution | None = None
    generation: int = 0
    consecutive_failures: int = 0
    all_solutions: list[Solution] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    ladder_rung_idx: int = 0  # current position on optimization ladder
    current_rung_name: str = ""  # name of the current ladder rung

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    def record(self, solution: Solution, eval_result: EvalResult, analysis: str = ""):
        self.timeline.append({
            "generation": solution.generation,
            "id": solution.id,
            "status": eval_result.status,
            "correct": eval_result.correct,
            "latency_us": eval_result.median_latency_us,
            "strategy": solution.strategy,
            "elapsed_s": self.elapsed_s,
            "analysis": analysis[:200],
        })

    def save_report(self, path: str | Path):
        """Save evolution report as JSON."""
        report = {
            "kernel": self.spec.name,
            "generations": self.generation,
            "best_latency_us": self.best_latency_us if self.best_latency_us != float("inf") else None,
            "best_solution_id": self.best_solution.id if self.best_solution else None,
            "elapsed_s": self.elapsed_s,
            "total_solutions": len(self.all_solutions),
            "timeline": self.timeline,
        }
        Path(path).write_text(json.dumps(report, indent=2))


class EvolutionLoop:
    """
    The main evolution loop.

    Algorithm:
    1. Generate initial kernel from spec
    2. Evaluate (compile → correctness → benchmark)
    3. If error: fix (up to N retries)
    4. If correct: analyze performance bottleneck
    5. Generate improved kernel based on analysis
    6. If improved: accept as new baseline
    7. Repeat until target or budget exhausted
    """

    def __init__(
        self,
        agent: KernelAgent,
        evaluator: KernelEvaluator,
        config: EvolutionConfig | None = None,
    ):
        self.agent = agent
        self.evaluator = evaluator
        self.config = config or EvolutionConfig()

        # Strategy learning (#5)
        self.strategy_db = None
        if self.config.enable_learning:
            try:
                from .eval.tournament import StrategyDatabase
                db_path = self.config.strategy_db_path
                self.strategy_db = StrategyDatabase(db_path)
                if self.strategy_db.records:
                    log.info(f"Strategy DB: {len(self.strategy_db.records)} records loaded")
            except Exception as e:
                log.debug(f"Strategy DB init failed: {e}")

    @classmethod
    def create(
        cls,
        backend: LLMBackend | None = None,
        config: EvolutionConfig | None = None,
    ) -> "EvolutionLoop":
        """Factory with sensible defaults."""
        config = config or EvolutionConfig()
        backend = backend or AnthropicBackend()
        agent = KernelAgent(backend=backend, gpu=config.gpu)
        evaluator = KernelEvaluator(config=config.eval_config)
        return cls(agent=agent, evaluator=evaluator, config=config)

    def run(self, spec: KernelSpec) -> EvolutionState:
        """Run the full evolution loop."""
        state = EvolutionState(spec=spec)
        output_dir = Path(self.config.output_dir) / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)

        self._print_header(spec)

        # Phase 1: Generate initial kernel
        console.print("\n[bold blue]Phase 1: Initial Generation[/]")

        # Inject reference corpus (#3) into the agent's context
        if self.config.use_corpus:
            try:
                from .agent.corpus import get_references, references_to_prompt, fetch_fla_references
                refs = get_references(spec.kernel_type)
                if self.config.fetch_fla_refs:
                    refs.extend(fetch_fla_references())
                if refs:
                    corpus_context = references_to_prompt(refs[:2])
                    console.print(f"  [dim]Loaded {len(refs)} reference kernels as few-shot examples[/]")
                    # Inject into spec context (agent will see it in the prompt)
                    spec._corpus_context = corpus_context
            except Exception as e:
                log.debug(f"Corpus loading failed: {e}")

        # Inject strategy history (#5) into the agent's context
        if self.strategy_db and self.strategy_db.records:
            strategy_context = self.strategy_db.to_prompt_context(spec.kernel_type)
            if strategy_context:
                console.print(f"  [dim]Loaded strategy history ({len(self.strategy_db.records)} records)[/]")
                spec._strategy_context = strategy_context

        # Inject adversarial test context so the agent knows edge cases upfront
        adversarial_suite = None
        try:
            from .eval.adversarial import generate_test_suite
            adversarial_suite = generate_test_suite(spec.kernel_type)
            adversarial_context = adversarial_suite.to_prompt_context()
            spec._adversarial_context = adversarial_context
            console.print(f"  [dim]Loaded adversarial test suite ({len(adversarial_suite.cases)} edge cases)[/]")
        except Exception as e:
            log.debug(f"Adversarial suite loading failed: {e}")

        with console.status("Generating initial kernel..."):
            solution = self.agent.generate_initial(spec)

        state.current_solution = solution
        state.all_solutions.append(solution)

        if self.config.save_all_solutions:
            solution.save(output_dir / f"gen{solution.generation:04d}_{solution.id}")

        # Phase 2: Evaluate and evolve
        console.print("\n[bold blue]Phase 2: Evolution Loop[/]")

        for gen in range(self.config.max_generations):
            state.generation = gen
            solution = state.current_solution

            # Evaluate
            console.print(f"\n[dim]Gen {gen}[/] Evaluating {solution.name}...")
            eval_result = self.evaluator.evaluate(spec, solution)
            solution.eval_results.append(eval_result)
            self.agent.update_history(solution.id, eval_result)

            # Display result
            self._print_eval_result(gen, solution, eval_result)

            # Handle errors with retries
            if not eval_result.passed:
                fix_attempts = 0
                while not eval_result.passed and fix_attempts < self.config.max_fix_attempts:
                    fix_attempts += 1
                    console.print(f"  [yellow]Fix attempt {fix_attempts}/{self.config.max_fix_attempts}[/]")

                    with console.status("Fixing..."):
                        solution = self.agent.improve(spec, solution, eval_result)
                    state.all_solutions.append(solution)

                    eval_result = self.evaluator.evaluate(spec, solution)
                    solution.eval_results.append(eval_result)
                    self.agent.update_history(solution.id, eval_result)
                    self._print_eval_result(gen, solution, eval_result, prefix="  ")

                if not eval_result.passed:
                    state.consecutive_failures += 1
                    state.record(solution, eval_result)

                    if state.consecutive_failures >= self.config.max_consecutive_failures:
                        console.print(f"\n[red]Stopping: {state.consecutive_failures} consecutive failures[/]")
                        break
                    continue

            # Reset failure counter on success
            state.consecutive_failures = 0
            state.current_solution = solution

            # Track best — require minimum improvement to accept
            latency = eval_result.median_latency_us or float("inf")
            if latency < state.best_latency_us:
                improvement = (state.best_latency_us - latency) / state.best_latency_us if state.best_latency_us != float("inf") else 1.0
                if improvement >= self.config.improvement_threshold or state.best_solution is None:
                    # Adversarial validation gate: verify on edge cases before accepting
                    adversarial_passed = True
                    if adversarial_suite is not None:
                        try:
                            adversarial_shapes = adversarial_suite.shapes_only()
                            adv_result = self.evaluator.evaluate(spec, solution, test_shapes=adversarial_shapes)
                            if not adv_result.correct:
                                adversarial_passed = False
                                adv_error = adv_result.runtime_error or adv_result.compile_error or "incorrect output"
                                console.print(f"  [yellow]⚠ Adversarial check failed: {adv_error[:120]}[/]")
                                console.print(f"  [dim]Keeping previous best (adversarial regression)[/]")
                        except Exception as e:
                            log.debug(f"Adversarial validation skipped: {e}")

                    if adversarial_passed:
                        state.best_latency_us = latency
                        state.best_solution = solution
                        console.print(f"  [green]★ New best! {latency:.1f} μs ({improvement:.1%} improvement)[/]")

                        if self.config.save_all_solutions:
                            solution.save(output_dir / f"best_gen{gen:04d}_{solution.id}")
                else:
                    console.print(f"  [dim]Marginal improvement ({improvement:.1%} < {self.config.improvement_threshold:.0%} threshold), keeping previous best[/]")

            # Check target
            if self.config.target_latency_us and latency <= self.config.target_latency_us:
                console.print(f"\n[bold green]🎯 Target reached! {latency:.1f} μs <= {self.config.target_latency_us} μs[/]")
                state.record(solution, eval_result)
                break

            # ISA analysis + performance analysis
            analysis = ""

            # Always run static ISA analysis (free, no GPU needed)
            try:
                from .eval.isa_analyzer import enrich_eval_with_isa
                isa_feedback = enrich_eval_with_isa(
                    solution.main_source, eval_result, spec.kernel_type
                )
                analysis += isa_feedback
                console.print(f"  [dim]ISA: {isa_feedback.split(chr(10))[0][:100]}[/]")
            except Exception as e:
                log.debug(f"ISA analysis skipped: {e}")

            # LLM-powered deep analysis periodically
            if gen > 0 and gen % self.config.analyze_every_n == 0:
                console.print(f"  [dim]Deep performance analysis...[/]")
                with console.status("Running analysis..."):
                    llm_analysis = self.agent.analyze_performance(spec, solution, eval_result)
                analysis += "\n\n## LLM Performance Analysis\n" + llm_analysis
                console.print(f"  [dim]Analysis: {llm_analysis[:150]}...[/]")

            state.record(solution, eval_result, analysis)

            # Advance optimization ladder
            try:
                from .agent.ladder import get_rung_for_generation, rung_to_prompt, get_ladder
                ladder = get_ladder(spec.name)
                rung, new_rung_idx = get_rung_for_generation(
                    spec.name, gen, eval_result.correct, state.ladder_rung_idx
                )
                if new_rung_idx > state.ladder_rung_idx:
                    console.print(f"  [cyan]⬆ Ladder: advancing to rung {new_rung_idx+1}/{len(ladder)}: {rung.name}[/]")
                state.ladder_rung_idx = new_rung_idx
                state.current_rung_name = rung.name
                ladder_context = rung_to_prompt(rung, new_rung_idx, len(ladder))
                analysis += "\n\n" + ladder_context
            except Exception as e:
                log.debug(f"Ladder skipped: {e}")

            # Generate next candidate (tournament or single)
            if gen < self.config.max_generations - 1:
                prev_eval = eval_result  # save for strategy recording

                if self.config.tournament_size > 1:
                    # Tournament mode (#4): generate multiple candidates, keep best
                    try:
                        from .eval.tournament import run_tournament
                        console.print(f"  [dim]Tournament: generating {self.config.tournament_size} candidates...[/]")
                        tournament = run_tournament(
                            agent=self.agent,
                            evaluator=self.evaluator,
                            spec=spec,
                            parent=solution,
                            parent_eval=eval_result,
                            analysis=analysis,
                            num_candidates=self.config.tournament_size,
                            parallel=self.config.tournament_parallel,
                        )
                        console.print(f"  [dim]{tournament.summary()}[/]")
                        next_solution = tournament.winner
                        next_eval = tournament.winner_eval
                    except Exception as e:
                        log.warning(f"Tournament failed, falling back to single candidate: {e}")
                        with console.status("Generating improved kernel..."):
                            next_solution = self.agent.improve(spec, solution, eval_result, analysis)
                        next_eval = None
                else:
                    with console.status("Generating improved kernel..."):
                        next_solution = self.agent.improve(spec, solution, eval_result, analysis)
                    next_eval = None

                # Record strategy outcome (#5)
                if self.strategy_db:
                    try:
                        record_eval = next_eval  # use tournament eval if available
                        if not record_eval:
                            record_eval = self.evaluator.evaluate(spec, next_solution) if next_eval is None else next_eval
                        self.strategy_db.record(
                            strategy=next_solution.strategy,
                            kernel_type=spec.kernel_type,
                            generation=gen,
                            parent_eval=prev_eval,
                            result_eval=record_eval,
                            ladder_rung=getattr(state, 'current_rung_name', ''),
                        )
                    except Exception as e:
                        log.debug(f"Strategy recording failed: {e}")

                state.current_solution = next_solution
                state.all_solutions.append(next_solution)

                if self.config.save_all_solutions:
                    next_solution.save(output_dir / f"gen{next_solution.generation:04d}_{next_solution.id}")

        # Final report
        self._print_summary(state)
        state.save_report(output_dir / "evolution_report.json")

        # Save strategy DB
        if self.strategy_db:
            self.strategy_db.save()
            console.print(f"  [dim]Strategy DB: {len(self.strategy_db.records)} records saved[/]")

        # Save best solution in starter-kit format
        if state.best_solution:
            starter_kit_dir = output_dir / "submission"
            self._export_starter_kit(state.best_solution, spec, starter_kit_dir)

        return state

    def _export_starter_kit(self, solution: Solution, spec: KernelSpec, output_dir: Path):
        """Export best solution in FlashInfer-Bench starter kit format."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # config.toml
        config_toml = f"""[solution]
name = "{solution.name}"
definition = "{spec.name}"
author = "{solution.author}"

[build]
language = "{solution.language}"
entry_point = "{solution.entry_point}"
"""
        (output_dir / "config.toml").write_text(config_toml)

        # Source files
        sol_dir = output_dir / "solution" / solution.language
        sol_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in solution.sources.items():
            (sol_dir / filename).write_text(content)

        console.print(f"\n[green]Submission exported to {output_dir}[/]")

    def _print_header(self, spec: KernelSpec):
        console.print(Panel(
            f"[bold]{spec.name}[/]\n"
            f"Type: {spec.kernel_type}\n"
            f"Axes: {', '.join(f'{a.name}={a.value}' if a.value else a.name for a in spec.axes)}\n"
            f"GPU: {self.config.gpu.name}\n"
            f"Max generations: {self.config.max_generations}",
            title="🔥 KernForge Evolution",
            border_style="blue",
        ))

    def _print_eval_result(self, gen: int, solution: Solution, result: EvalResult, prefix: str = ""):
        status_emoji = {
            "passed": "✅",
            "compile_error": "🔴",
            "runtime_error": "💥",
            "incorrect": "❌",
        }
        emoji = status_emoji.get(result.status, "?")
        latency_str = f"{result.median_latency_us:.1f}μs" if result.median_latency_us else "N/A"
        error_str = ""
        if result.compile_error:
            error_str = f" | {result.compile_error[:80]}"
        elif result.runtime_error:
            error_str = f" | {result.runtime_error[:80]}"

        console.print(f"{prefix}{emoji} [{result.status}] {latency_str} | {solution.strategy[:60]}{error_str}")

    def _print_summary(self, state: EvolutionState):
        table = Table(title="\n🏁 Evolution Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Kernel", state.spec.name)
        table.add_row("Generations", str(state.generation + 1))
        table.add_row("Total solutions", str(len(state.all_solutions)))
        table.add_row("Elapsed", f"{state.elapsed_s:.1f}s")

        passed = sum(1 for t in state.timeline if t.get("correct"))
        table.add_row("Correct solutions", f"{passed}/{len(state.timeline)}")

        if state.best_latency_us != float("inf"):
            table.add_row("Best latency", f"{state.best_latency_us:.1f} μs")
            table.add_row("Best solution", state.best_solution.id if state.best_solution else "N/A")

        console.print(table)
