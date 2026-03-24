"""
KernForge main entry point.
Drop-in compatible with the official mlsys26-agent-baseline CLI interface,
enhanced with static analysis, domain knowledge, tournament selection,
and cross-run strategy learning.

Usage:
  # Same as baseline:
  python -m kernforge.main --config config/config_iterative.yaml

  # With KernForge enhancements (default):
  python -m kernforge.main --config config/config_iterative.yaml \
      --use_static_check --use_domain_knowledge --use_tournament 3
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from typing import cast

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Imports from baseline-compatible modules ───

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_dataset_roots():
    return {
        "flashinfer-trace": os.path.join(REPO_TOP_PATH, "datasets", "flashinfer-trace"),
        "mlsys26-contest": os.path.join(REPO_TOP_PATH, "datasets", "mlsys26-contest"),
    }


def _get_dataset_root(test_source: str) -> str:
    roots = _get_dataset_roots()
    if test_source not in roots:
        raise ValueError(f"Unknown test_source: {test_source}")
    return roots[test_source]


def _require_yaml():
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyYAML is required for kernforge.main config loading/dumping. "
            "Install project dependencies to use the legacy runner."
        ) from e
    return yaml


# ─── Task loading (baseline compatible) ───


def load_definition(op_type: str, problem_name: str, dataset_root: str) -> dict:
    path = os.path.join(dataset_root, "definitions", op_type, f"{problem_name}.json")
    with open(path, "r") as f:
        return json.load(f)


def load_tasks(tasks_path: str, test_source: str) -> list[dict]:
    dataset_root = _get_dataset_root(test_source)
    with open(tasks_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    tasks = []
    for line in lines:
        parts = line.split(" ", 1)
        level = parts[0]
        if len(parts) == 1:
            op_dir = os.path.join(dataset_root, "definitions", level)
            if os.path.isdir(op_dir):
                problems = sorted(
                    f[:-5] for f in os.listdir(op_dir) if f.endswith(".json")
                )
            else:
                problems = []
        else:
            problems = [p.strip() for p in parts[1].split(",") if p.strip()]
        tasks.extend({"level": level, "problem_id": str(p)} for p in problems)
    return tasks


# ─── API client (baseline compatible) ───


def create_inference_server(api_type: str):
    if api_type == "openai":
        import openai

        return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif api_type in ("claude", "anthropic"):
        import anthropic

        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")


def query_llm(
    server,
    model_name: str,
    prompt: str,
    max_tokens: int = 16384,
    retry_times: int = 5,
    **kwargs,
):
    import random
    import anthropic as _anthropic

    kwargs.setdefault("temperature", 0.6)

    is_anthropic = isinstance(server, _anthropic.Anthropic)

    for attempt in range(retry_times):
        try:
            if is_anthropic:
                response = server.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                return "".join(b.text for b in response.content if hasattr(b, "text"))
            else:
                response = server.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    **kwargs,
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{retry_times}): {e}"
            )
            if attempt == retry_times - 1:
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            time.sleep(wait)


# ─── Code extraction (baseline compatible) ───

import re


def extract_first_code(output: str) -> str:
    trimmed = output.strip()
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        for prefix in ("python", "cpp", "triton"):
            if code.startswith(prefix):
                code = code[len(prefix) :].strip()
        return code
    return output


def extract_edits(output: str) -> list[tuple[str, str]]:
    edits = []
    # Try regex-based extraction (handles tags inside code blocks)
    for m in re.finditer(r"<old_str_(\d+)>(.*?)</old_str_\1>", output, re.DOTALL):
        idx = m.group(1)
        raw_old = m.group(2)
        # Accept both </new_str_N> and </old_str_N> as closing tag (common LLM mistake)
        new_match = re.search(
            rf"<new_str_{idx}>(.*?)</(?:new_str|old_str)_{idx}>",
            output,
            re.DOTALL,
        )
        if new_match:
            edits.append((raw_old, new_match.group(1)))
    if edits:
        return edits
    # Fallback: line-based parsing
    for line in output.split("\n"):
        if line.strip().startswith("<old_str_"):
            try:
                idx = int(line.strip().split("_")[2].split(">")[0])
                raw_old = output.split(f"<old_str_{idx}>")[1].split(
                    f"</old_str_{idx}>"
                )[0]
                raw_new = output.split(f"<new_str_{idx}>")[1].split(
                    f"</new_str_{idx}>"
                )[0]
                edits.append((raw_old, raw_new))
            except Exception:
                continue
    return edits


def clean_edit_markers(code: str) -> str:
    """Strip any residual XML edit markers from kernel code."""
    # Remove <reasoning_N>...</reasoning_N> blocks (may span multiple lines)
    code = re.sub(r"<reasoning_\d+>.*?</reasoning_\d+>", "", code, flags=re.DOTALL)
    # Remove unclosed <reasoning_N> blocks (to end of file or next tag)
    code = re.sub(
        r"<reasoning_\d+>.*?(?=<(?:old_str|new_str)_\d+>|$)", "", code, flags=re.DOTALL
    )
    # Remove individual tags
    code = re.sub(r"</?(old_str|new_str|reasoning)_\d+>", "", code)
    # Clean up excessive blank lines left behind
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code


def str_replace(content: str, old: str, new: str) -> str:
    if old not in content:
        old_stripped = old.strip()
        new = new.strip()
        if old_stripped in content:
            return content.replace(old_stripped, new, 1)
        logger.warning(f"str_replace: old_str not found in content")
        return content
    if content.count(old) > 1:
        logger.warning(f"str_replace: multiple occurrences, skipping")
        return content
    return content.replace(old, new, 1)


# ─── Kernel type detection ───


def detect_kernel_type(level: str, problem_id: str) -> str:
    """Detect kernel type from level/problem_id for type-specific prompting."""
    level_lower = level.lower()
    pid_lower = problem_id.lower()
    combined = f"{level_lower}_{pid_lower}"

    if "gdn" in combined:
        return "gdn"
    if "moe" in combined:
        return "moe"
    if "gemm" in combined:
        return "gemm"
    if "dsa" in combined or "paged" in combined:
        return "dsa_paged"
    return "general"


# ─── Core agent loop ───


def run_agent(args, inference_server, level, problem_id):
    """Run agent on a single problem."""
    from kernforge.eval.flashinfer_eval import EvalResult, calculate_score, eval_kernel

    result_save_path = os.path.join(args.save_path, f"{level}_{problem_id}")
    os.makedirs(result_save_path, exist_ok=True)

    dataset_root = _get_dataset_root(args.test_source)
    definition = load_definition(level, problem_id, dataset_root)

    kernel_type = detect_kernel_type(level, problem_id)

    task_params = {
        "definition": json.dumps(definition, indent=4),
        "target_gpu": args.gpu_name,
        "gpu_name": args.gpu_name,
        "gpu_architecture": args.gpu_architecture,
        "dtype_str": str(definition.get("inputs", "unknown")),
    }

    with open(os.path.join(result_save_path, "definition.json"), "w") as f:
        f.write(json.dumps(definition, indent=4))

    # Load strategy DB if enabled
    strategy_db = None
    if getattr(args, "use_strategy_db", False):
        try:
            from kernforge.eval.tournament import StrategyDatabase

            strategy_db = StrategyDatabase()
        except Exception as e:
            logger.debug(f"Strategy DB not available: {e}")

    # Load corpus context if enabled
    corpus_context = None
    if getattr(args, "use_domain_knowledge", False):
        try:
            from kernforge.agent.corpus import get_references, references_to_prompt

            refs = get_references(kernel_type)
            if refs:
                corpus_context = references_to_prompt(refs)
        except Exception as e:
            logger.debug(f"Corpus not available: {e}")

    # Create eval function
    eval_fn = args.eval_fn

    # ─── Dispatch to agent type ───
    agent_type = getattr(args, "agent_type", "iterative")

    if agent_type == "hybrid":
        from kernforge.agent.hybrid_agent import run_hybrid_loop

        best_kernel, best_metric = run_hybrid_loop(
            task_params=task_params,
            inference_server=inference_server,
            args=args,
            eval_fn=eval_fn,
            kernel_type=kernel_type,
            strategy_db=strategy_db,
            corpus_context=corpus_context or "",
            log_path=result_save_path,
            dataset_root=dataset_root,
        )
    elif agent_type == "evolve":
        from kernforge.agent.hybrid_agent import run_hybrid_loop

        # "evolve" is explore-only mode: reuse the hybrid loop with no exploit phase.
        original_fraction = getattr(args, "explore_fraction", 0.4)
        args.explore_fraction = 1.0
        try:
            best_kernel, best_metric = run_hybrid_loop(
                task_params=task_params,
                inference_server=inference_server,
                args=args,
                eval_fn=eval_fn,
                kernel_type=kernel_type,
                strategy_db=strategy_db,
                corpus_context=corpus_context or "",
                log_path=result_save_path,
                dataset_root=dataset_root,
            )
        finally:
            args.explore_fraction = original_fraction
    else:
        # Default: iterative agent (propose once, then refine)
        best_kernel, best_metric = _run_iterative_agent(
            task_params=task_params,
            inference_server=inference_server,
            args=args,
            eval_fn=eval_fn,
            kernel_type=kernel_type,
            strategy_db=strategy_db,
            corpus_context=corpus_context,
            log_path=result_save_path,
            dataset_root=dataset_root,
            problem_id=problem_id,
        )

    # Save best
    if best_kernel:
        with open(
            os.path.join(result_save_path, f"global_best_kernel_{args.total_steps}.py"),
            "w",
        ) as f:
            f.write(best_kernel)
        with open(
            os.path.join(
                result_save_path, f"global_best_metrics_{args.total_steps}.json"
            ),
            "w",
        ) as f:
            json.dump(best_metric.model_dump(), f, indent=4)

    return best_kernel, best_metric


def _run_iterative_agent(
    task_params,
    inference_server,
    args,
    eval_fn,
    kernel_type,
    strategy_db,
    corpus_context,
    log_path,
    dataset_root,
    problem_id,
):
    """Iterative agent: propose once, then refine with str_replace edits."""
    from kernforge.eval.flashinfer_eval import calculate_score

    previous_kernels = []
    previous_metrics = []
    best_kernel = None
    best_metric = None
    best_score = (0, 0, 0)

    level = getattr(args, "level", "")

    for step in tqdm(range(args.total_steps), desc=f"{level}_{problem_id}"):
        try:
            if len(previous_kernels) == 0:
                from kernforge.prompt.proposer_prompt import (
                    generate_proposer_prompt,
                    generate_pool_prompt,
                )

                pool_prompt = generate_pool_prompt(kernel_pool=[], metrics_pool=[])
                prompt = generate_proposer_prompt(
                    task_params=task_params,
                    pool_prompt=pool_prompt,
                    kernel_type=kernel_type,
                    gpu_name=args.gpu_name,
                    strategy_db=strategy_db,
                    corpus_context=corpus_context,
                )
                output = (
                    query_llm(
                        inference_server,
                        args.model_name,
                        prompt,
                        args.max_completion_tokens,
                    )
                    or ""
                )
                kernel = extract_first_code(output)
                metrics = eval_fn(kernel, problem_id, dataset_root)
                _save_step(log_path, f"proposal_0_{step + 1}", kernel, metrics, prompt)
            else:
                from kernforge.prompt.tuner_prompt import generate_tuner_prompt

                # KernForge: NCU profiling for correct-but-slow kernels
                ncu_profile = None
                last_m = previous_metrics[-1] if previous_metrics else None
                if (
                    last_m
                    and hasattr(last_m, "correct")
                    and last_m.correct
                    and last_m.speedup < 1.5
                    and step % 3 == 0
                ):  # Profile every 3rd step to save time
                    try:
                        from kernforge.eval.agents_integration import profile_kernel_ncu

                        ncu_profile = profile_kernel_ncu(
                            previous_kernels[-1], problem_id, dataset_root
                        )
                        if ncu_profile:
                            logger.info(
                                f"  NCU: bottleneck={ncu_profile.bottleneck}, "
                                f"DRAM={ncu_profile.dram_throughput_pct:.0f}%, "
                                f"SM={ncu_profile.sm_throughput_pct:.0f}%"
                            )
                    except Exception as e:
                        logger.debug(f"NCU profiling skipped: {e}")

                prompt = generate_tuner_prompt(
                    previous_kernels=previous_kernels[-args.max_memory_round :],
                    previous_metrics=previous_metrics[-args.max_memory_round :],
                    task_params=task_params,
                    ncu_profile=ncu_profile,
                )
                output = (
                    query_llm(
                        inference_server,
                        args.model_name,
                        prompt,
                        args.max_completion_tokens,
                    )
                    or ""
                )
                kernel = previous_kernels[-1]
                edits = extract_edits(output)
                if edits:
                    applied = 0
                    for old, new in edits:
                        new_kernel = str_replace(kernel, old, new)
                        if new_kernel != kernel:
                            kernel = new_kernel
                            applied += 1
                    if applied == 0:
                        logger.warning(
                            f"All {len(edits)} edits failed to match — trying full code extraction"
                        )
                        extracted = extract_first_code(output)
                        if (
                            extracted
                            and extracted != output
                            and not re.search(
                                r"</?(?:old_str|new_str|reasoning)_\d+>", extracted
                            )
                        ):
                            kernel = extracted
                        else:
                            logger.warning(
                                "Full extraction also has markers — keeping previous kernel"
                            )
                else:
                    extracted = extract_first_code(output)
                    if extracted and extracted != output:
                        # Reject if it contains edit markers (LLM mixed code with edit tags)
                        if re.search(
                            r"</?(?:old_str|new_str|reasoning)_\d+>", extracted
                        ):
                            logger.warning(
                                "Extracted code contains edit markers — trying to parse edits from code block"
                            )
                            code_edits = extract_edits(extracted)
                            if code_edits:
                                for old, new in code_edits:
                                    new_kernel = str_replace(kernel, old, new)
                                    if new_kernel != kernel:
                                        kernel = new_kernel
                            else:
                                logger.warning(
                                    "Cannot parse edits from code block — keeping previous kernel"
                                )
                        else:
                            kernel = extracted
                # Safety: strip any residual edit markers
                kernel = clean_edit_markers(kernel)
                # Final validation: if kernel doesn't compile, revert to previous
                try:
                    compile(kernel, "<kernel>", "exec")
                except SyntaxError:
                    logger.warning(
                        "Kernel has syntax error after edit — reverting to previous kernel"
                    )
                    kernel = previous_kernels[-1]
                metrics = eval_fn(kernel, problem_id, dataset_root)
                _save_step(log_path, f"tune_0_{step + 1}", kernel, metrics, prompt)

            previous_kernels.append(kernel)
            previous_metrics.append(metrics)
            if len(previous_kernels) > args.max_memory_round:
                previous_kernels.pop(0)
                previous_metrics.pop(0)

            score = calculate_score(metrics)
            if score > best_score:
                best_score = score
                best_kernel = kernel
                best_metric = metrics
                logger.info(
                    f"  Step {step + 1}: New best — compiled={metrics.compiled}, "
                    f"correct={metrics.correct}, speedup={metrics.speedup:.3f}"
                )
        except Exception as e:
            from kernforge.eval.flashinfer_eval import ModalAppStopped

            if isinstance(e, ModalAppStopped):
                logger.error(
                    f"Modal app stopped at step {step + 1} — aborting remaining steps"
                )
                break
            raise

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


# ─── Main orchestration ───


def run_main_loop(args):
    """Run the agent on all tasks and report results."""
    tasks = load_tasks(args.tasks_path, args.test_source)
    inference_server = create_inference_server(api_type=args.api_type)

    correct_count = 0
    sum_speedup = 0.0
    total_count = len(tasks)

    for task in tqdm(tasks, desc="Processing problems"):
        level, problem_id = task["level"], task["problem_id"]

        # Skip already-completed problems
        result_path = os.path.join(args.save_path, f"{level}_{problem_id}")
        kernel_file = os.path.join(
            result_path, f"global_best_kernel_{args.total_steps}.py"
        )
        metrics_file = os.path.join(
            result_path, f"global_best_metrics_{args.total_steps}.json"
        )
        if os.path.exists(kernel_file) and os.path.exists(metrics_file):
            from kernforge.eval.flashinfer_eval import read_metrics

            cached = cast(tuple[bool, float], read_metrics(metrics_file))
            correctness, speedup = cached
            if correctness:
                correct_count += 1
                sum_speedup += speedup
            print(
                f"Cached: {level}_{problem_id} — Correct: {correctness}, Speedup: {speedup:.4f}"
            )
            continue

        try:
            best_kernel, best_metrics = run_agent(
                args, inference_server, level, problem_id
            )
            if best_metrics and best_metrics.correct:
                correct_count += 1
                sum_speedup += best_metrics.speedup
            print(
                f"Completed: {level}_{problem_id} — "
                f"Correct: {best_metrics.correct if best_metrics else False}, "
                f"Speedup: {best_metrics.speedup if best_metrics else 0:.4f}"
            )
        except Exception as e:
            print(f"Failed: {level}_{problem_id} — {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(
        f"Results: {correct_count}/{total_count} correct, Sum speedup: {sum_speedup:.4f}"
    )
    if total_count > 0:
        print(f"Average speedup: {sum_speedup / total_count:.4f}")
        print(f"Accuracy: {correct_count / total_count:.2%}")


def load_config_from_yaml(args, parser, argv=None):
    if args.config is None:
        return args
    yaml = _require_yaml()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    parser.set_defaults(**{k: v for k, v in config_dict.items() if hasattr(args, k)})
    return parser.parse_args(argv)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="KernForge: AI Kernel Generation Agent"
    )

    # ─── Baseline-compatible args ───
    parser.add_argument(
        "--test_source",
        type=str,
        default="mlsys26-contest",
        choices=["mlsys26-contest", "flashinfer-trace"],
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="hybrid",
        choices=["iterative", "evolve", "hybrid"],
    )
    parser.add_argument("--tasks_path", type=str, default="config/tasks_all.txt")
    parser.add_argument("--gpu_name", type=str, default="B200")
    parser.add_argument("--gpu_architecture", type=str, default="Blackwell")
    parser.add_argument(
        "--api_type", type=str, default="claude", choices=["openai", "claude"]
    )
    parser.add_argument("--model_name", type=str, default="claude-sonnet-4-5")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--total_steps", type=int, default=25)
    parser.add_argument("--max_memory_round", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument(
        "--eval_backend", type=str, default="local", choices=["local", "modal"]
    )
    parser.add_argument("--modal_gpu", type=str, default="B200")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--explore_fraction",
        type=float,
        default=0.4,
        help="Fraction of total_steps for explore phase in hybrid agent",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    # ─── KernForge-specific args ───
    parser.add_argument(
        "--use_static_check",
        action="store_true",
        default=True,
        help="Pre-filter kernels with static analysis before GPU eval",
    )
    parser.add_argument(
        "--use_domain_knowledge",
        action="store_true",
        default=True,
        help="Inject GPU kernel domain knowledge into prompts",
    )
    parser.add_argument(
        "--use_strategy_db",
        action="store_true",
        default=True,
        help="Use cross-run strategy learning",
    )
    parser.add_argument(
        "--no_enhancements",
        action="store_true",
        help="Disable all KernForge enhancements (pure baseline mode)",
    )

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    args = load_config_from_yaml(args, parser, argv)

    if args.no_enhancements:
        args.use_static_check = False
        args.use_domain_knowledge = False
        args.use_strategy_db = False

    start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    if args.save_path is None:
        args.save_path = os.path.join(
            REPO_TOP_PATH,
            "outputs",
            f"kernforge_{args.agent_type}_{args.test_source}_{args.total_steps}_{start_time}",
        )

    os.makedirs(args.save_path, exist_ok=True)
    yaml = _require_yaml()
    with open(os.path.join(args.save_path, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Create eval function
    if args.eval_backend == "modal":
        import modal
        from kernforge.eval.modal_eval import create_modal_app, ensure_dataset_synced

        modal_app, remote_eval_fn, dataset_vol = create_modal_app(args.modal_gpu)

        from kernforge.eval.flashinfer_eval import create_eval_fn

        args.eval_fn = create_eval_fn(
            "modal",
            args.test_source,
            remote_fn=remote_eval_fn,
            use_static_check=args.use_static_check,
        )
        with modal.enable_output(), modal_app.run():
            ensure_dataset_synced(
                dataset_vol,
                _get_dataset_root(args.test_source),
                args.test_source,
            )
            run_main_loop(args)
    else:
        from kernforge.eval.flashinfer_eval import create_eval_fn

        args.eval_fn = create_eval_fn(
            "local",
            use_static_check=args.use_static_check,
        )
        run_main_loop(args)


if __name__ == "__main__":
    main()
