"""
Solution representation for generated kernels.
Tracks source code, build config, evaluation results, and evolution lineage.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class EvalResult:
    """Result of evaluating a kernel solution."""
    correct: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    median_latency_us: float | None = None
    min_latency_us: float | None = None
    throughput_tflops: float | None = None
    compile_error: str | None = None
    runtime_error: str | None = None
    ncu_metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.correct and self.compile_error is None and self.runtime_error is None

    @property
    def status(self) -> str:
        if self.compile_error:
            return "compile_error"
        if self.runtime_error:
            return "runtime_error"
        if not self.correct:
            return "incorrect"
        return "passed"


@dataclass
class Solution:
    """A kernel solution with its source code and evaluation history."""

    name: str
    definition: str  # kernel definition name
    language: Literal["triton", "cuda", "python"]
    entry_point: str  # function name or file::function

    sources: dict[str, str]  # filename -> source code
    dependencies: list[str] = field(default_factory=list)

    # Metadata
    author: str = "kernforge-agent"
    description: str = ""
    generation: int = 0  # evolution generation
    parent_id: str | None = None  # hash of parent solution

    # Results
    eval_results: list[EvalResult] = field(default_factory=list)

    # Agent reasoning trace
    strategy: str = ""  # what optimization strategy was attempted
    reasoning: str = ""  # agent's reasoning for this solution

    @property
    def id(self) -> str:
        """Deterministic hash of the solution source code."""
        content = json.dumps(self.sources, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def best_result(self) -> EvalResult | None:
        passed = [r for r in self.eval_results if r.passed]
        if not passed:
            return None
        return min(passed, key=lambda r: r.median_latency_us or float("inf"))

    @property
    def main_source(self) -> str:
        """Get the primary source file content."""
        if self.language == "triton":
            return self.sources.get("kernel.py", next(iter(self.sources.values()), ""))
        elif self.language == "cuda":
            return self.sources.get("kernel.cu", next(iter(self.sources.values()), ""))
        return next(iter(self.sources.values()), "")

    def to_flashinfer_bench_json(self) -> dict[str, Any]:
        """Export to FlashInfer-Bench solution JSON format."""
        return {
            "name": self.name,
            "definition": self.definition,
            "description": self.description,
            "author": self.author,
            "spec": {
                "language": self.language,
                "target_hardware": ["NVIDIA_B200"],
                "dependencies": self.dependencies,
                "entry_point": self.entry_point,
            },
            "sources": [
                {"path": path, "content": content}
                for path, content in self.sources.items()
            ],
        }

    def save(self, output_dir: str | Path):
        """Save solution files to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write source files
        for filename, content in self.sources.items():
            filepath = output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)

        # Write solution.json
        with open(output_dir / "solution.json", "w") as f:
            json.dump(self.to_flashinfer_bench_json(), f, indent=2)

        # Write evolution metadata
        meta = {
            "id": self.id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "strategy": self.strategy,
            "reasoning": self.reasoning,
            "eval_results": [
                {
                    "status": r.status,
                    "correct": r.correct,
                    "median_latency_us": r.median_latency_us,
                    "max_abs_error": r.max_abs_error,
                    "timestamp": r.timestamp,
                }
                for r in self.eval_results
            ],
        }
        with open(output_dir / "evolution.json", "w") as f:
            json.dump(meta, f, indent=2)
