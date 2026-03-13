"""
Tournament Selection + Strategy Learning.

#4: Multi-candidate tournament — generate 3-5 variants per generation,
evaluate all, keep the best. Genetic algorithm without the crossover (yet).

#5: Cross-generation learning — persist what strategies worked into a
database. On future runs, the agent starts with proven strategies.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..kernel.solution import EvalResult, Solution
from ..kernel.spec import KernelSpec

log = logging.getLogger(__name__)

# ============================================================================
# #4: Tournament Selection
# ============================================================================

@dataclass
class TournamentResult:
    """Result of a multi-candidate tournament."""
    candidates: list[tuple[Solution, EvalResult]]  # all (solution, eval) pairs
    winner: Solution
    winner_eval: EvalResult
    generation: int
    elapsed_s: float = 0.0

    @property
    def num_correct(self) -> int:
        return sum(1 for _, er in self.candidates if er.correct)

    @property
    def num_total(self) -> int:
        return len(self.candidates)

    def summary(self) -> str:
        lines = [f"Tournament gen {self.generation}: {self.num_correct}/{self.num_total} correct"]
        for i, (sol, er) in enumerate(self.candidates):
            marker = "★" if sol.id == self.winner.id else " "
            lat = f"{er.median_latency_us:.1f}μs" if er.median_latency_us else er.status
            lines.append(f"  {marker} [{i+1}] {sol.strategy[:40]:40s} → {lat}")
        return "\n".join(lines)


def run_tournament(
    agent,
    evaluator,
    spec: KernelSpec,
    parent: Solution,
    parent_eval: EvalResult,
    analysis: str,
    num_candidates: int = 3,
    parallel: bool = False,
) -> TournamentResult:
    """
    Generate multiple candidate improvements, evaluate all, return the best.

    If parallel=True, uses threads to generate candidates simultaneously
    (LLM API calls are I/O bound, so threading helps).
    """
    start = time.time()
    candidates: list[tuple[Solution, EvalResult]] = []

    if parallel and num_candidates > 1:
        # Generate candidates in parallel (I/O bound LLM calls)
        solutions = _generate_parallel(agent, spec, parent, parent_eval, analysis, num_candidates)
    else:
        # Generate sequentially with varied strategies
        solutions = _generate_sequential(agent, spec, parent, parent_eval, analysis, num_candidates)

    # Evaluate all candidates
    for sol in solutions:
        try:
            eval_result = evaluator.evaluate(spec, sol)
            sol.eval_results.append(eval_result)
            candidates.append((sol, eval_result))
        except Exception as e:
            log.warning(f"Evaluation failed for candidate {sol.id}: {e}")
            candidates.append((sol, EvalResult(correct=False, runtime_error=str(e))))

    # Select winner: fastest correct kernel, or least-broken if none correct
    correct = [(s, e) for s, e in candidates if e.correct and e.median_latency_us]
    if correct:
        winner_sol, winner_eval = min(correct, key=lambda x: x[1].median_latency_us)
    elif any(e.correct for _, e in candidates):
        # Correct but no latency (benchmark failed)
        winner_sol, winner_eval = next((s, e) for s, e in candidates if e.correct)
    else:
        # None correct — pick the one with smallest error
        winner_sol, winner_eval = min(
            candidates,
            key=lambda x: (
                0 if x[1].compile_error is None else 1,
                0 if x[1].runtime_error is None else 1,
                x[1].max_abs_error or float("inf"),
            ),
        )

    # Crossover: if we have a winner + runner-up (both correct), try combining them
    if len(correct) >= 2:
        sorted_correct = sorted(correct, key=lambda x: x[1].median_latency_us)
        runner_up_sol, runner_up_eval = sorted_correct[1]
        try:
            crossover_sol = agent.crossover(
                spec, winner_sol, winner_eval, runner_up_sol, runner_up_eval
            )
            crossover_eval = evaluator.evaluate(spec, crossover_sol)
            crossover_sol.eval_results.append(crossover_eval)
            candidates.append((crossover_sol, crossover_eval))

            # If crossover beats the winner, use it instead
            if (crossover_eval.correct and crossover_eval.median_latency_us
                    and crossover_eval.median_latency_us < winner_eval.median_latency_us):
                log.info(
                    f"Crossover beat winner: {crossover_eval.median_latency_us:.1f}μs "
                    f"< {winner_eval.median_latency_us:.1f}μs"
                )
                winner_sol = crossover_sol
                winner_eval = crossover_eval
        except Exception as e:
            log.debug(f"Crossover failed: {e}")

    return TournamentResult(
        candidates=candidates,
        winner=winner_sol,
        winner_eval=winner_eval,
        generation=parent.generation + 1,
        elapsed_s=time.time() - start,
    )


def _generate_parallel(agent, spec, parent, parent_eval, analysis, n) -> list[Solution]:
    """Generate N candidates in parallel using threads."""
    solutions = []

    def _gen(i):
        # Vary the temperature/approach for diversity
        modified_analysis = analysis + f"\n\n[Candidate {i+1}/{n}: try a DIFFERENT approach than other candidates]"
        return agent.improve(spec, parent, parent_eval, modified_analysis)

    with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
        futures = [executor.submit(_gen, i) for i in range(n)]
        for future in as_completed(futures):
            try:
                solutions.append(future.result())
            except Exception as e:
                log.warning(f"Candidate generation failed: {e}")

    return solutions


def _generate_sequential(agent, spec, parent, parent_eval, analysis, n) -> list[Solution]:
    """Generate N candidates sequentially with varied strategy hints."""
    solutions = []
    strategy_hints = [
        "Focus on MEMORY optimization: reduce HBM traffic, improve coalescing.",
        "Focus on COMPUTE optimization: use tensor cores, reduce redundant ops.",
        "Focus on OCCUPANCY: adjust tile sizes to reduce register pressure.",
        "Focus on FUSION: combine separate passes into one loop over the data.",
        "Try a DIFFERENT ALGORITHM: restructure the computation entirely.",
    ]

    for i in range(n):
        hint = strategy_hints[i % len(strategy_hints)]
        modified_analysis = analysis + f"\n\n## Strategy Directive\n{hint}"
        try:
            sol = agent.improve(spec, parent, parent_eval, modified_analysis)
            solutions.append(sol)
        except Exception as e:
            log.warning(f"Candidate {i+1} generation failed: {e}")

    return solutions


# ============================================================================
# #5: Cross-Generation Strategy Learning
# ============================================================================

@dataclass
class StrategyRecord:
    """A record of a strategy attempt and its outcome."""
    strategy: str
    kernel_type: str
    generation: int
    parent_latency_us: float | None
    result_latency_us: float | None
    correct: bool
    improvement_pct: float | None  # positive = faster
    error_type: str | None = None
    ladder_rung: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def succeeded(self) -> bool:
        return self.correct and (self.improvement_pct or 0) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "kernel_type": self.kernel_type,
            "generation": self.generation,
            "parent_latency_us": self.parent_latency_us,
            "result_latency_us": self.result_latency_us,
            "correct": self.correct,
            "improvement_pct": self.improvement_pct,
            "error_type": self.error_type,
            "ladder_rung": self.ladder_rung,
            "timestamp": self.timestamp,
        }


class StrategyDatabase:
    """
    Persistent database of strategy outcomes across evolution runs.

    Learns which optimization strategies work for which kernel types,
    at which stages of optimization, and with what expected improvement.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else Path.home() / ".kernforge" / "strategy_db.json"
        self.records: list[StrategyRecord] = []
        self._load()

    def _load(self):
        """Load existing records from disk."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self.records = [
                    StrategyRecord(**r) for r in data.get("records", [])
                ]
                log.info(f"Loaded {len(self.records)} strategy records from {self.db_path}")
            except (json.JSONDecodeError, TypeError) as e:
                log.warning(f"Failed to load strategy DB: {e}")

    def save(self):
        """Persist records to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"records": [r.to_dict() for r in self.records]}
        self.db_path.write_text(json.dumps(data, indent=2))

    def record(
        self,
        strategy: str,
        kernel_type: str,
        generation: int,
        parent_eval: EvalResult | None,
        result_eval: EvalResult,
        ladder_rung: str = "",
    ):
        """Record a strategy attempt and its outcome."""
        parent_lat = parent_eval.median_latency_us if parent_eval else None
        result_lat = result_eval.median_latency_us

        improvement = None
        if parent_lat and result_lat and parent_lat > 0:
            improvement = (parent_lat - result_lat) / parent_lat * 100  # positive = faster

        record = StrategyRecord(
            strategy=strategy,
            kernel_type=kernel_type,
            generation=generation,
            parent_latency_us=parent_lat,
            result_latency_us=result_lat,
            correct=result_eval.correct,
            improvement_pct=improvement,
            error_type=result_eval.status if not result_eval.correct else None,
            ladder_rung=ladder_rung,
        )
        self.records.append(record)
        self.save()

    def get_best_strategies(
        self,
        kernel_type: str,
        n: int = 5,
        ladder_rung: str | None = None,
    ) -> list[StrategyRecord]:
        """Get the most successful strategies for a kernel type."""
        relevant = [r for r in self.records
                    if r.kernel_type == kernel_type and r.succeeded]

        if ladder_rung:
            rung_specific = [r for r in relevant if r.ladder_rung == ladder_rung]
            if rung_specific:
                relevant = rung_specific

        # Sort by improvement
        relevant.sort(key=lambda r: -(r.improvement_pct or 0))
        return relevant[:n]

    def get_failed_strategies(
        self,
        kernel_type: str,
        n: int = 5,
    ) -> list[StrategyRecord]:
        """Get strategies that consistently fail — avoid these."""
        failed = [r for r in self.records
                  if r.kernel_type == kernel_type and not r.succeeded]

        # Group by strategy name and count failures
        fail_counts: dict[str, int] = {}
        for r in failed:
            key = r.strategy[:50]
            fail_counts[key] = fail_counts.get(key, 0) + 1

        # Return most-failed strategies
        sorted_fails = sorted(fail_counts.items(), key=lambda x: -x[1])
        result = []
        for strategy, count in sorted_fails[:n]:
            matching = [r for r in failed if r.strategy[:50] == strategy]
            result.append(matching[0])
        return result

    def to_prompt_context(self, kernel_type: str) -> str:
        """Format learned strategies as LLM context."""
        best = self.get_best_strategies(kernel_type, n=5)
        failed = self.get_failed_strategies(kernel_type, n=3)

        if not best and not failed:
            return ""

        lines = ["## Strategy History (from previous optimization runs)"]

        if best:
            lines.append("\n### ✅ Strategies that worked well:")
            for r in best:
                lines.append(
                    f"- \"{r.strategy[:60]}\" → {r.improvement_pct:.1f}% faster "
                    f"(gen {r.generation}, {r.result_latency_us:.1f}μs)"
                )

        if failed:
            lines.append("\n### ❌ Strategies that failed (AVOID):")
            for r in failed:
                lines.append(
                    f"- \"{r.strategy[:60]}\" → {r.error_type or 'regression'}"
                )

        return "\n".join(lines)

    @property
    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        if not self.records:
            return {"total": 0}

        succeeded = [r for r in self.records if r.succeeded]
        return {
            "total": len(self.records),
            "succeeded": len(succeeded),
            "success_rate": len(succeeded) / len(self.records),
            "avg_improvement": (
                sum(r.improvement_pct for r in succeeded if r.improvement_pct) / len(succeeded)
                if succeeded else 0
            ),
            "kernel_types": list(set(r.kernel_type for r in self.records)),
        }
