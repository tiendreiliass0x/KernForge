"""
Tournament selection: generate N diverse candidates per step, keep the best.

Instead of the baseline's single-candidate-per-step approach, we:
1. Generate N (default 3) candidates with diverse strategy hints
2. Evaluate all N (can be parallel on Modal)
3. Keep the best by calculate_score()
4. Feed the winner into the next round

This is ~3x more candidates per eval budget with minimal extra LLM cost.
"""

import logging
from typing import Callable, Optional

from kernforge.eval.flashinfer_eval import EvalResult, calculate_score

logger = logging.getLogger(__name__)


# Strategy hints for diverse exploration
STRATEGY_HINTS = [
    None,  # No hint — baseline approach
    "Focus on CORRECTNESS. Use simple patterns with proper masks on every tl.load/tl.store.",
    "Focus on MEMORY. Fuse operations to minimize HBM round-trips. Use register tiling.",
    "Focus on COMPUTE. Use tl.dot() for every matmul. Add @triton.autotune with 6+ configs.",
    "Focus on ALGORITHMIC CHANGE. Can you reformulate the computation for better parallelism?",
    "Focus on FUSION. Combine the entire forward pass into one kernel with one global read/write.",
    "Focus on OCCUPANCY. Use smaller tile sizes and more warps to maximize SM utilization.",
]


def tournament_propose(
    prompt_fn: Callable,
    llm_fn: Callable,
    eval_fn: Callable,
    extract_fn: Callable,
    *,
    n_candidates: int = 3,
    problem_id: str = "",
    dataset_root: str = "",
    strategy_hints: list[str] = None,
) -> tuple[str, EvalResult, list[tuple[str, EvalResult]]]:
    """
    Generate N diverse candidates and return the best one.

    Args:
        prompt_fn: Callable(strategy_hint) -> prompt string
        llm_fn: Callable(prompt) -> LLM output string
        eval_fn: Callable(kernel_code, task_id, dataset_root) -> EvalResult
        extract_fn: Callable(llm_output) -> kernel code string
        n_candidates: Number of candidates to generate
        problem_id: Task ID for eval
        dataset_root: Dataset root path
        strategy_hints: Optional custom strategy hints

    Returns:
        (best_kernel, best_metrics, all_results)
    """
    hints = strategy_hints or STRATEGY_HINTS
    all_results = []
    best_kernel = None
    best_metric = None
    best_score = (0, 0, 0)

    for i in range(n_candidates):
        hint = hints[i % len(hints)]

        try:
            prompt = prompt_fn(hint)
            output = llm_fn(prompt)
            kernel = extract_fn(output)
            metrics = eval_fn(kernel, problem_id, dataset_root)
        except Exception as e:
            logger.warning(f"Tournament candidate {i+1}/{n_candidates} failed: {e}")
            kernel = ""
            metrics = EvalResult(error=str(e))

        all_results.append((kernel, metrics))
        score = calculate_score(metrics)

        if score > best_score:
            best_score = score
            best_kernel = kernel
            best_metric = metrics
            logger.debug(
                f"  Tournament candidate {i+1}/{n_candidates}: "
                f"NEW BEST — compiled={metrics.compiled}, "
                f"correct={metrics.correct}, speedup={metrics.speedup:.3f}"
            )
        else:
            logger.debug(
                f"  Tournament candidate {i+1}/{n_candidates}: "
                f"score={score} (best={best_score})"
            )

    winner_idx = next(
        i for i, (k, m) in enumerate(all_results)
        if k == best_kernel and m == best_metric
    )
    logger.info(
        f"  Tournament: {n_candidates} candidates, "
        f"winner #{winner_idx+1} — "
        f"compiled={best_metric.compiled}, correct={best_metric.correct}, "
        f"speedup={best_metric.speedup:.3f}"
    )

    return best_kernel, best_metric, all_results


def inject_strategy_hint(prompt: str, hint: Optional[str]) -> str:
    """Inject a strategy hint into a prompt."""
    if not hint:
        return prompt

    marker = "Generate the complete, runnable implementation:"
    if marker in prompt:
        return prompt.replace(
            marker,
            f"\n## Strategy Focus\n{hint}\n\n{marker}",
        )
    # Fallback: append at the end
    return prompt + f"\n\n## Strategy Focus\n{hint}\n"
