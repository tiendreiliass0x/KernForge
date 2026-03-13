"""
Hybrid agent: evolutionary exploration → iterative exploitation.

Strategy:
1. Explore phase (evolve): Generate diverse candidates, build elite pool
2. Exploit phase (iterative): Take best candidate, refine with str_replace edits

Phase transition happens when:
- We find a correct kernel with speedup > 1.0, OR
- We've used explore_fraction of total steps

This combines the baseline's two agents into a single, smarter agent.
"""

import argparse
import json
import logging
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

from kernforge.eval.flashinfer_eval import EvalResult, calculate_score

logger = logging.getLogger(__name__)


def run_hybrid_loop(
    task_params: dict,
    inference_server,
    args: argparse.Namespace,
    eval_fn,
    *,
    kernel_type: str = "general",
    strategy_db=None,
    corpus_context: str = None,
    log_path: str = None,
    dataset_root: str = None,
) -> tuple[str, EvalResult]:
    """
    Run the hybrid evolve→iterative agent loop.

    Phase 1 (explore): Generate diverse candidates using different strategies
    Phase 2 (exploit): Refine the best candidate with str_replace edits
    """
    from kernforge.main import extract_first_code, extract_edits, str_replace, query_llm
    from kernforge.prompt.proposer_prompt import generate_proposer_prompt, generate_pool_prompt
    from kernforge.prompt.tuner_prompt import generate_tuner_prompt

    total_steps = args.total_steps
    explore_fraction = getattr(args, "explore_fraction", 0.4)
    explore_steps = max(2, int(total_steps * explore_fraction))
    exploit_steps = total_steps - explore_steps

    problem_id = getattr(args, "problem_id", "unknown")
    level = getattr(args, "level", "unknown")

    # ─── Phase 1: Explore ───
    logger.info(f"[Hybrid] Phase 1: Explore ({explore_steps} steps)")

    kernel_pool = []
    metrics_pool = []
    elite_pool = []  # (kernel, metrics, score) sorted by score
    best_kernel = None
    best_metric = None
    best_score = (0, 0, 0)

    # Strategy hints for diverse exploration
    strategy_hints = [
        None,  # First proposal: no hint, baseline approach
        "Focus on CORRECTNESS first. Use simple, straightforward Triton patterns.",
        "Focus on MEMORY OPTIMIZATION. Minimize HBM traffic, fuse operations.",
        "Focus on COMPUTE OPTIMIZATION. Use tl.dot() and tensor cores aggressively.",
        "Try a DIFFERENT ALGORITHM entirely. Think about the problem from scratch.",
        "Focus on AUTOTUNE. Add @triton.autotune with many configurations.",
        "Focus on FUSION. Combine multiple operations into a single kernel.",
    ]

    for step in tqdm(range(explore_steps), desc=f"Explore {level}_{problem_id}"):
        hint_idx = step % len(strategy_hints)
        strategy_hint = strategy_hints[hint_idx]

        # Build prompt with strategy hint
        pool_prompt = generate_pool_prompt(
            kernel_pool=kernel_pool[-3:],  # Recent context
            metrics_pool=metrics_pool[-3:],
            elite_kernel_pool=[k for k, _, _ in elite_pool[:2]],
            elite_metrics_pool=[m for _, m, _ in elite_pool[:2]],
        )

        # Inject strategy hint into prompt
        extra_context = ""
        if strategy_hint:
            extra_context = f"\n## Strategy Focus\n{strategy_hint}\n"

        prompt = generate_proposer_prompt(
            task_params=task_params,
            pool_prompt=pool_prompt,
            kernel_type=kernel_type,
            gpu_name=args.gpu_name,
            strategy_db=strategy_db,
            corpus_context=corpus_context,
        )
        if extra_context:
            prompt = prompt.replace(
                "Generate the complete, runnable implementation:",
                extra_context + "\nGenerate the complete, runnable implementation:",
            )

        output = query_llm(inference_server, args.model_name, prompt,
                           args.max_completion_tokens)
        kernel = extract_first_code(output)
        metrics = eval_fn(kernel, problem_id, dataset_root)

        # Log
        _save_step(log_path, f"explore_{step+1}", kernel, metrics, prompt)

        kernel_pool.append(kernel)
        metrics_pool.append(metrics)

        score = calculate_score(metrics)
        elite_pool.append((kernel, metrics, score))
        elite_pool.sort(key=lambda x: x[2], reverse=True)
        elite_pool = elite_pool[:5]  # Keep top 5

        if score > best_score:
            best_score = score
            best_kernel = kernel
            best_metric = metrics
            logger.info(
                f"  Explore step {step+1}: New best — "
                f"compiled={metrics.compiled}, correct={metrics.correct}, "
                f"speedup={metrics.speedup:.3f}"
            )

        # Early transition: if we have a correct kernel with good speedup
        if metrics.correct and metrics.speedup > 1.0 and step >= 2:
            remaining_explore = explore_steps - step - 1
            exploit_steps += remaining_explore  # Give remaining steps to exploit
            logger.info(
                f"  Early transition at step {step+1}: "
                f"speedup={metrics.speedup:.3f} > 1.0, "
                f"transferring {remaining_explore} steps to exploit phase"
            )
            break

    if best_kernel is None:
        logger.warning("[Hybrid] No kernels generated in explore phase!")
        return "", EvalResult(error="No kernels generated")

    # ─── Phase 2: Exploit ───
    logger.info(f"[Hybrid] Phase 2: Exploit ({exploit_steps} steps) from speedup={best_metric.speedup:.3f}")

    # Start with the best kernel from explore phase
    previous_kernels = [best_kernel]
    previous_metrics = [best_metric]
    max_memory = getattr(args, "max_memory_round", 5)

    # Optional: NCU profile the best kernel for targeted optimization
    ncu_context = None
    if best_metric.correct and best_metric.speedup < 2.0:
        try:
            from kernforge.eval.profiling import run_ncu_profile, format_ncu_for_prompt
            ncu_result = run_ncu_profile(best_kernel, problem_id, dataset_root)
            if not ncu_result.get("error"):
                ncu_context = format_ncu_for_prompt(ncu_result)
                logger.info(f"  NCU bottleneck: {ncu_result.get('bottleneck', 'unknown')}")
        except Exception as e:
            logger.debug(f"  NCU profiling skipped: {e}")

    for step in tqdm(range(exploit_steps), desc=f"Exploit {level}_{problem_id}"):
        # Build tuner prompt with NCU feedback
        prompt = generate_tuner_prompt(
            previous_kernels=previous_kernels[-max_memory:],
            previous_metrics=previous_metrics[-max_memory:],
            task_params=task_params,
        )

        # Inject NCU context if available
        if ncu_context and step == 0:
            prompt = prompt.replace(
                "### Previous Kernels and Metrics:",
                ncu_context + "\n\n### Previous Kernels and Metrics:",
            )

        output = query_llm(inference_server, args.model_name, prompt,
                           args.max_completion_tokens)

        # Apply str_replace edits
        kernel = previous_kernels[-1]
        edits = extract_edits(output)
        if edits:
            for old, new in edits:
                kernel = str_replace(kernel, old, new)
        else:
            extracted = extract_first_code(output)
            if extracted and extracted != output:
                kernel = extracted

        metrics = eval_fn(kernel, problem_id, dataset_root)

        _save_step(log_path, f"exploit_{explore_steps + step + 1}", kernel, metrics, prompt)

        previous_kernels.append(kernel)
        previous_metrics.append(metrics)
        if len(previous_kernels) > max_memory:
            previous_kernels.pop(0)
            previous_metrics.pop(0)

        score = calculate_score(metrics)
        if score > best_score:
            best_score = score
            best_kernel = kernel
            best_metric = metrics
            logger.info(
                f"  Exploit step {step+1}: New best — "
                f"compiled={metrics.compiled}, correct={metrics.correct}, "
                f"speedup={metrics.speedup:.3f}"
            )

            # Re-profile after significant improvement
            if metrics.correct and metrics.speedup > 1.5 and ncu_context is None:
                try:
                    from kernforge.eval.profiling import run_ncu_profile, format_ncu_for_prompt
                    ncu_result = run_ncu_profile(best_kernel, problem_id, dataset_root)
                    if not ncu_result.get("error"):
                        ncu_context = format_ncu_for_prompt(ncu_result)
                except Exception:
                    pass

    # Record strategies if DB available
    if strategy_db:
        try:
            strategy_db.record(
                strategy=f"hybrid_{explore_steps}explore_{exploit_steps}exploit",
                kernel_type=kernel_type,
                generation=total_steps,
                parent_eval=None,
                result_eval=best_metric,
            )
            strategy_db.save()
        except Exception:
            pass

    return best_kernel, best_metric


def _save_step(log_path, prefix, kernel, metrics, prompt):
    if log_path is None:
        return
    with open(os.path.join(log_path, f"{prefix}.py"), "w") as f:
        f.write(kernel)
    with open(os.path.join(log_path, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics.model_dump(), f)
    with open(os.path.join(log_path, f"{prefix}_prompt.txt"), "w") as f:
        f.write(prompt)
