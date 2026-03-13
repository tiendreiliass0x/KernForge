"""
Core LLM agent for kernel generation and improvement.
Supports Anthropic Claude API with extensible backend interface.
"""
from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..kernel.hardware import B200, GPUSpec
from ..kernel.solution import EvalResult, Solution
from ..kernel.spec import KernelSpec
from . import prompts

log = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Request to generate or improve a kernel."""
    kernel_spec: KernelSpec
    gpu: GPUSpec = field(default_factory=lambda: B200)
    current_solution: Solution | None = None
    eval_result: EvalResult | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    analysis: str = ""
    mode: str = "generate"  # generate | improve | fix


@dataclass
class GenerationResult:
    """Result from the LLM agent."""
    source_code: str
    strategy: str = ""
    reasoning: str = ""
    raw_response: str = ""
    model: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @abstractmethod
    def generate(self, system: str, user: str, max_tokens: int = 8192) -> GenerationResult:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def name(self) -> str:
        return f"anthropic/{self.model}"

    def generate(self, system: str, user: str, max_tokens: int = 8192) -> GenerationResult:
        t0 = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed = time.time() - t0

        raw = response.content[0].text
        source_code = _extract_python_code(raw)
        strategy = _extract_strategy(raw)

        return GenerationResult(
            source_code=source_code,
            strategy=strategy,
            reasoning=raw,
            raw_response=raw,
            model=self.model,
            latency_s=elapsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible API backend (for local models, Together, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def name(self) -> str:
        return f"openai-compat/{self.model}"

    def generate(self, system: str, user: str, max_tokens: int = 8192) -> GenerationResult:
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        elapsed = time.time() - t0

        raw = response.choices[0].message.content
        source_code = _extract_python_code(raw)
        strategy = _extract_strategy(raw)

        return GenerationResult(
            source_code=source_code,
            strategy=strategy,
            raw_response=raw,
            model=self.model,
            latency_s=elapsed,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )


class KernelAgent:
    """
    Main agent that generates and evolves GPU kernels.

    The agent maintains a history of attempts and uses structured prompts
    to generate increasingly optimized kernel implementations.
    """

    def __init__(self, backend: LLMBackend, gpu: GPUSpec = B200):
        self.backend = backend
        self.gpu = gpu
        self.history: list[dict[str, Any]] = []

    def generate_initial(self, spec: KernelSpec) -> Solution:
        """Generate the first kernel implementation from a spec."""
        log.info(f"Generating initial kernel for {spec.name} using {self.backend.name()}")

        # Use domain-enriched system prompt based on kernel type
        try:
            from .domain_knowledge import get_enhanced_system_prompt
            system_prompt = get_enhanced_system_prompt(spec.kernel_type)
        except ImportError:
            system_prompt = prompts.SYSTEM_PROMPT

        user_prompt = prompts.GENERATE_FROM_REFERENCE_PROMPT.format(
            kernel_spec=spec.to_prompt_context(),
            hardware_spec=self.gpu.to_prompt_context(),
        )

        result = self.backend.generate(
            system=system_prompt,
            user=user_prompt,
        )

        solution = Solution(
            name=f"{spec.name}-gen0-{result.model.split('/')[-1][:8]}",
            definition=spec.name,
            language="triton",
            entry_point="kernel",
            sources={"kernel.py": result.source_code},
            dependencies=["triton >= 2.3", "torch"],
            author=f"kernforge/{self.backend.name()}",
            description=f"Initial generation by {self.backend.name()}",
            generation=0,
            strategy=result.strategy,
            reasoning=result.reasoning,
        )

        self._record_attempt(spec, solution, result)
        return solution

    def improve(self, spec: KernelSpec, current: Solution, eval_result: EvalResult,
                analysis: str = "") -> Solution:
        """Generate an improved version of an existing kernel."""
        log.info(f"Improving kernel {current.name} (gen {current.generation}) — status: {eval_result.status}")

        # Use domain-enriched system prompt
        try:
            from .domain_knowledge import get_enhanced_system_prompt, get_mutation_strategy
            system_prompt = get_enhanced_system_prompt(spec.kernel_type)
            # Get structured mutation strategy
            is_mem_bound = None
            if eval_result.ncu_metrics:
                bw = eval_result.ncu_metrics.get("achieved_bandwidth_gb_s")
                if bw is not None:
                    is_mem_bound = bw / 8000 > 0.5  # >50% bandwidth = memory bound
            mutation = get_mutation_strategy(
                current.generation, eval_result.status, is_mem_bound
            )
            mutation_context = f"\n## Mutation Strategy: {mutation['name']}\n{mutation.get('instruction', mutation.get('description', ''))}\n"
        except ImportError:
            system_prompt = prompts.SYSTEM_PROMPT
            mutation_context = ""

        # Build status description
        if eval_result.compile_error:
            status_desc = f"fails to compile with error:\n{eval_result.compile_error}"
            mode = "fix"
        elif eval_result.runtime_error:
            status_desc = f"crashes at runtime with error:\n{eval_result.runtime_error}"
            mode = "fix"
        elif not eval_result.correct:
            status_desc = (
                f"produces incorrect results "
                f"(max_abs_error={eval_result.max_abs_error:.6e}, "
                f"max_rel_error={eval_result.max_rel_error:.6e})"
            )
            mode = "fix"
        else:
            status_desc = (
                f"is correct but needs to be faster "
                f"(median latency: {eval_result.median_latency_us:.1f} μs)"
            )
            mode = "improve"

        # Build history summary
        history_lines = []
        for h in self.history[-5:]:  # last 5 attempts
            hist_status = h.get("status", "unknown")
            hist_strategy = h.get("strategy", "none")
            hist_latency = h.get("latency_us", "N/A")
            history_lines.append(f"  - Gen {h.get('generation', '?')}: {hist_status} | {hist_strategy} | {hist_latency}μs")

        # Build performance summary
        perf_lines = []
        if eval_result.median_latency_us:
            perf_lines.append(f"Median latency: {eval_result.median_latency_us:.1f} μs")
        if eval_result.ncu_metrics:
            for k, v in eval_result.ncu_metrics.items():
                perf_lines.append(f"{k}: {v}")

        if mode == "fix":
            user_prompt = prompts.FIX_ERROR_PROMPT.format(
                error_type=eval_result.status,
                error_message=eval_result.compile_error or eval_result.runtime_error or "Incorrect output",
                current_code=current.main_source,
                kernel_spec=spec.to_prompt_context(),
            )
        else:
            user_prompt = prompts.IMPROVE_KERNEL_PROMPT.format(
                status_description=status_desc,
                current_code=current.main_source,
                kernel_spec=spec.to_prompt_context(),
                hardware_spec=self.gpu.to_prompt_context(),
                attempt_history="\n".join(history_lines) or "None",
                performance_summary="\n".join(perf_lines) or "No profiling data available",
                analysis=(analysis or "No detailed analysis available.") + mutation_context,
            )

        result = self.backend.generate(
            system=system_prompt,
            user=user_prompt,
        )

        new_gen = current.generation + 1
        solution = Solution(
            name=f"{spec.name}-gen{new_gen}-{current.id[:6]}",
            definition=spec.name,
            language="triton",
            entry_point="kernel",
            sources={"kernel.py": result.source_code},
            dependencies=["triton >= 2.3", "torch"],
            author=f"kernforge/{self.backend.name()}",
            description=f"Gen {new_gen}: {result.strategy}",
            generation=new_gen,
            parent_id=current.id,
            strategy=result.strategy,
            reasoning=result.reasoning,
        )

        self._record_attempt(spec, solution, result)
        return solution

    def crossover(self, spec: KernelSpec,
                  parent_a: Solution, eval_a: EvalResult,
                  parent_b: Solution, eval_b: EvalResult) -> Solution:
        """Generate a crossover kernel combining the best ideas from two parents."""
        log.info(f"Crossover: {parent_a.name} x {parent_b.name}")

        # Use domain-enriched system prompt
        try:
            from .domain_knowledge import get_enhanced_system_prompt
            system_prompt = get_enhanced_system_prompt(spec.kernel_type)
        except ImportError:
            system_prompt = prompts.SYSTEM_PROMPT

        parent_a_perf = f"{eval_a.median_latency_us:.1f} μs" if eval_a.median_latency_us else eval_a.status
        parent_b_perf = f"{eval_b.median_latency_us:.1f} μs" if eval_b.median_latency_us else eval_b.status

        user_prompt = prompts.CROSSOVER_PROMPT.format(
            parent_a_code=parent_a.main_source,
            parent_a_perf=parent_a_perf,
            parent_a_strategy=parent_a.strategy,
            parent_b_code=parent_b.main_source,
            parent_b_perf=parent_b_perf,
            parent_b_strategy=parent_b.strategy,
            kernel_spec=spec.to_prompt_context(),
            hardware_spec=self.gpu.to_prompt_context(),
        )

        result = self.backend.generate(
            system=system_prompt,
            user=user_prompt,
        )

        new_gen = max(parent_a.generation, parent_b.generation) + 1
        solution = Solution(
            name=f"{spec.name}-cross-gen{new_gen}-{parent_a.id[:4]}x{parent_b.id[:4]}",
            definition=spec.name,
            language="triton",
            entry_point="kernel",
            sources={"kernel.py": result.source_code},
            dependencies=["triton >= 2.3", "torch"],
            author=f"kernforge/{self.backend.name()}",
            description=f"Crossover gen {new_gen}: {parent_a.id[:6]} x {parent_b.id[:6]}",
            generation=new_gen,
            parent_id=parent_a.id,
            strategy=f"crossover: {result.strategy}",
            reasoning=result.reasoning,
        )

        self._record_attempt(spec, solution, result)
        return solution

    def analyze_performance(self, spec: KernelSpec, solution: Solution,
                            eval_result: EvalResult) -> str:
        """Ask the LLM to analyze performance bottlenecks."""
        user_prompt = prompts.ANALYZE_PERFORMANCE_PROMPT.format(
            current_code=solution.main_source,
            kernel_spec=spec.to_prompt_context(),
            hardware_spec=self.gpu.to_prompt_context(),
            benchmark_results=f"Median latency: {eval_result.median_latency_us or 'N/A'} μs",
            ncu_metrics="\n".join(f"{k}: {v}" for k, v in eval_result.ncu_metrics.items()) or "Not available",
        )

        result = self.backend.generate(
            system=prompts.SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=2048,
        )
        return result.raw_response

    def _record_attempt(self, spec: KernelSpec, solution: Solution, result: GenerationResult):
        self.history.append({
            "generation": solution.generation,
            "id": solution.id,
            "strategy": result.strategy,
            "status": "generated",
            "latency_us": None,
            "llm_latency_s": result.latency_s,
            "tokens": result.input_tokens + result.output_tokens,
        })

    def update_history(self, solution_id: str, eval_result: EvalResult):
        """Update history with evaluation results."""
        for h in reversed(self.history):
            if h.get("id") == solution_id:
                h["status"] = eval_result.status
                h["latency_us"] = eval_result.median_latency_us
                break


# --- Utilities ---

def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response."""
    # Try ```python ... ``` blocks first
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        # Return the longest match (usually the complete kernel)
        return max(matches, key=len).strip()

    # Try ``` ... ``` blocks
    matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Fallback: if the response looks like pure code, use it
    if "import triton" in text or "@triton.jit" in text:
        return text.strip()

    raise ValueError("Could not extract Python code from LLM response")


def _extract_strategy(text: str) -> str:
    """Extract strategy comment from LLM response."""
    match = re.search(r"<!--\s*STRATEGY:\s*(.*?)\s*-->", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try first line of non-code text
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("```") and not line.startswith("import"):
            return line[:200]
    return ""
