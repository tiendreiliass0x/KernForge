"""
Kernel specification parser for FlashInfer-Bench definitions.
Parses kernel definitions into structured specs that the agent can reason about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: list[str]  # symbolic shape expressions like ["batch_size", "seq_len", "num_heads", "head_size"]


@dataclass
class AxisSpec:
    name: str
    type: Literal["const", "var"]
    value: int | None = None  # only for const axes


@dataclass
class KernelSpec:
    """Complete specification of a kernel to generate."""

    name: str
    description: str
    kernel_type: str  # e.g., "gdn", "moe", "sparse_attention"

    axes: list[AxisSpec]
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    constraints: list[str]

    reference_code: str  # Python reference implementation

    # Derived
    const_axes: dict[str, int] = field(default_factory=dict)
    var_axes: list[str] = field(default_factory=list)

    # Injected context (set by evolution loop, not serialized)
    _corpus_context: str = ""
    _strategy_context: str = ""

    def __post_init__(self):
        self.const_axes = {a.name: a.value for a in self.axes if a.type == "const" and a.value is not None}
        self.var_axes = [a.name for a in self.axes if a.type == "var"]

    @classmethod
    def from_definition_json(cls, data: dict[str, Any]) -> "KernelSpec":
        """Parse from FlashInfer-Bench definition JSON format."""
        axes = []
        for ax in data.get("axes", []):
            axes.append(AxisSpec(
                name=ax["name"],
                type=ax["type"],
                value=ax.get("value"),
            ))

        inputs = []
        for inp in data.get("inputs", []):
            inputs.append(TensorSpec(
                name=inp["name"],
                dtype=inp["dtype"],
                shape=inp["shape"],
            ))

        outputs = []
        for out in data.get("outputs", []):
            outputs.append(TensorSpec(
                name=out["name"],
                dtype=out["dtype"],
                shape=out["shape"],
            ))

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            kernel_type=data.get("type", "unknown"),
            axes=axes,
            inputs=inputs,
            outputs=outputs,
            constraints=data.get("constraints", []),
            reference_code=data.get("reference", ""),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "KernelSpec":
        """Load from a definition JSON file."""
        with open(path) as f:
            return cls.from_definition_json(json.load(f))

    @classmethod
    def from_flashinfer_bench(cls, definition_name: str, dataset_path: str | Path) -> "KernelSpec":
        """Load from FlashInfer-Bench dataset by definition name."""
        dataset_path = Path(dataset_path)
        def_path = dataset_path / "definitions" / f"{definition_name}.json"
        if def_path.exists():
            return cls.from_file(def_path)

        # Support nested definitions/<op_type>/<definition>.json layouts used by the dataset.
        for f in (dataset_path / "definitions").rglob("*.json"):
            if f.stem == definition_name:
                return cls.from_file(f)
            with open(f) as fh:
                data = json.load(fh)
                if data.get("name") == definition_name:
                    return cls.from_definition_json(data)
        raise FileNotFoundError(f"Definition '{definition_name}' not found in {dataset_path}")

    def to_prompt_context(self) -> str:
        """Format spec as context string for LLM prompts."""
        lines = [
            f"# Kernel: {self.name}",
            f"# Type: {self.kernel_type}",
            f"# Description: {self.description}",
            "",
            "## Axes (dimensions)",
        ]
        for ax in self.axes:
            val = f" = {ax.value}" if ax.value is not None else " (variable)"
            lines.append(f"  - {ax.name}: {ax.type}{val}")

        lines.append("\n## Inputs")
        for inp in self.inputs:
            lines.append(f"  - {inp.name}: {inp.dtype} shape={inp.shape}")

        lines.append("\n## Outputs")
        for out in self.outputs:
            lines.append(f"  - {out.name}: {out.dtype} shape={out.shape}")

        if self.constraints:
            lines.append("\n## Constraints")
            for c in self.constraints:
                lines.append(f"  - {c}")

        lines.append("\n## Reference Implementation (Python)")
        lines.append("```python")
        lines.append(self.reference_code)
        lines.append("```")

        # Injected few-shot examples from corpus (#3)
        if self._corpus_context:
            lines.append("\n" + self._corpus_context)

        # Injected strategy history (#5)
        if self._strategy_context:
            lines.append("\n" + self._strategy_context)

        return "\n".join(lines)
