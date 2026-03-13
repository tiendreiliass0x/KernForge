"""
Static Analysis — catch bugs and missed optimizations before GPU execution.

Faster than running the kernel, cheaper than LLM calls. Run this on every
generated kernel BEFORE evaluation to give the agent immediate feedback
and save GPU time on obviously broken code.

Three levels:
1. Syntax/import validation (does it parse?)
2. Triton-specific pattern checks (common LLM bugs)
3. Optimization opportunity detection (what's left on the table?)
"""
from __future__ import annotations

import ast
import re
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class StaticIssue:
    """A single issue found by static analysis."""
    severity: str  # "error", "warning", "info"
    category: str  # "syntax", "correctness", "performance", "style"
    line: int | None
    message: str
    suggestion: str = ""

    def __str__(self):
        loc = f" (line {self.line})" if self.line else ""
        return f"[{self.severity.upper()}] {self.category}{loc}: {self.message}"


@dataclass
class StaticAnalysisResult:
    """Complete static analysis result."""
    issues: list[StaticIssue] = field(default_factory=list)
    metrics: dict[str, int | float] = field(default_factory=dict)
    valid_python: bool = True
    has_triton_jit: bool = False
    has_entry_point: bool = False
    triton_jit_lines: set = field(default_factory=set)

    @property
    def errors(self) -> list[StaticIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[StaticIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def blocking(self) -> bool:
        """Are there issues that will definitely prevent the kernel from running?"""
        return bool(self.errors)

    def to_prompt_context(self) -> str:
        """Format as feedback for the LLM."""
        if not self.issues:
            return "## Static Analysis: ✅ No issues detected"

        lines = ["## Static Analysis Results"]

        if self.errors:
            lines.append(f"\n### ❌ Errors ({len(self.errors)}) — MUST FIX before running")
            for issue in self.errors:
                lines.append(f"- Line {issue.line or '?'}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"  Fix: {issue.suggestion}")

        if self.warnings:
            lines.append(f"\n### ⚠️ Warnings ({len(self.warnings)}) — likely bugs or performance issues")
            for issue in self.warnings:
                lines.append(f"- Line {issue.line or '?'}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"  Suggestion: {issue.suggestion}")

        infos = [i for i in self.issues if i.severity == "info"]
        if infos:
            lines.append(f"\n### ℹ️ Optimization opportunities ({len(infos)})")
            for issue in infos:
                lines.append(f"- {issue.message}")

        if self.metrics:
            lines.append("\n### Code Metrics")
            for k, v in self.metrics.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)


def analyze(source: str, entry_point: str = "kernel",
            kernel_type: str = "general") -> StaticAnalysisResult:
    """Run full static analysis on kernel source."""
    result = StaticAnalysisResult()

    # Level 1: Python syntax
    _check_syntax(source, result)
    if not result.valid_python:
        return result  # can't do further analysis on broken syntax

    # Level 2: Triton-specific correctness
    # Entry point check FIRST — sets has_triton_jit flag used by pattern checks
    _check_entry_point(source, entry_point, result)
    _check_triton_patterns(source, result)

    # Level 3: Optimization opportunities
    _check_optimizations(source, result, kernel_type)

    # Collect metrics
    _collect_metrics(source, result)

    return result


def _check_syntax(source: str, result: StaticAnalysisResult):
    """Check if the source is valid Python."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        result.valid_python = False
        result.issues.append(StaticIssue(
            severity="error",
            category="syntax",
            line=e.lineno,
            message=f"SyntaxError: {e.msg}",
            suggestion="Fix the Python syntax error before proceeding.",
        ))


def _check_entry_point(source: str, entry_point: str, result: StaticAnalysisResult):
    """Check that the expected entry point function exists."""
    tree = ast.parse(source)

    func_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_names.append(node.name)

    if entry_point not in func_names:
        # Check common alternatives
        alternatives = [n for n in func_names if n in ("kernel", "run", "forward", "main")]
        if alternatives:
            result.issues.append(StaticIssue(
                severity="warning",
                category="correctness",
                line=None,
                message=f"Entry point '{entry_point}' not found. Found: {alternatives}",
                suggestion=f"Rename the function to '{entry_point}' or update the entry point.",
            ))
        else:
            result.issues.append(StaticIssue(
                severity="error",
                category="correctness",
                line=None,
                message=f"No entry point function found. Expected '{entry_point}'. Functions: {func_names}",
            ))
    else:
        result.has_entry_point = True

    # Check for @triton.jit decorator and record line ranges
    result.triton_jit_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                dec_str = ast.dump(decorator)
                if "triton" in dec_str and ("jit" in dec_str or "autotune" in dec_str):
                    result.has_triton_jit = True
                    # Record all lines inside this function body
                    for lineno in range(node.lineno, node.end_lineno + 1):
                        result.triton_jit_lines.add(lineno)

    if not result.has_triton_jit:
        result.issues.append(StaticIssue(
            severity="error",
            category="correctness",
            line=None,
            message="No @triton.jit or @triton.autotune decorated function found.",
            suggestion="The kernel function must be decorated with @triton.jit.",
        ))


def _check_triton_patterns(source: str, result: StaticAnalysisResult):
    """Check for common Triton bugs that LLMs make."""
    lines = source.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        # Bug: tl.load without mask on variable-dimension access
        if "tl.load(" in stripped and "mask" not in stripped:
            # Check if this is loading from a variable-size tensor
            # (not a constexpr-sized one)
            if "arange" in source[:source.index(stripped)] or "offs" in stripped.lower():
                result.issues.append(StaticIssue(
                    severity="warning",
                    category="correctness",
                    line=i,
                    message="tl.load() without mask — will crash on boundary tiles",
                    suggestion="Add mask=..., other=0.0 for variable-size dimensions",
                ))

        # Bug: tl.store without mask
        if "tl.store(" in stripped and "mask" not in stripped:
            if "arange" in source or "offs" in stripped.lower():
                result.issues.append(StaticIssue(
                    severity="warning",
                    category="correctness",
                    line=i,
                    message="tl.store() without mask — may write out of bounds",
                    suggestion="Add mask= parameter for variable-size dimensions",
                ))

        # Bug: accumulating in bf16
        if "tl.zeros" in stripped and "bfloat16" in stripped:
            # Check if this looks like an accumulator (followed by +=)
            context = "\n".join(lines[i:i+5])
            if "+=" in context or "sum" in context:
                result.issues.append(StaticIssue(
                    severity="warning",
                    category="correctness",
                    line=i,
                    message="Accumulator initialized as bfloat16 — will cause numerical drift",
                    suggestion="Use tl.zeros([...], dtype=tl.float32) for accumulators",
                ))

        # Bug: torch operations inside triton kernel (only inside @triton.jit body)
        in_jit = hasattr(result, 'triton_jit_lines') and i in result.triton_jit_lines
        if in_jit and any(f"torch.{op}" in stripped for op in ["sigmoid", "softplus", "exp", "sum", "matmul"]):
            result.issues.append(StaticIssue(
                severity="error",
                category="correctness",
                line=i,
                message=f"torch.* operation inside Triton kernel — must use tl.* equivalents",
                suggestion="Replace torch.sigmoid with tl.sigmoid, torch.exp with tl.exp, etc.",
            ))

        # Bug: numpy inside triton kernel (only inside @triton.jit body)
        if "np." in stripped or "numpy" in stripped:
            if in_jit:
                result.issues.append(StaticIssue(
                    severity="error",
                    category="correctness",
                    line=i,
                    message="NumPy operation inside Triton kernel — not supported",
                ))

        # Bug: using .item() inside kernel
        if ".item()" in stripped:
            result.issues.append(StaticIssue(
                severity="error",
                category="correctness",
                line=i,
                message=".item() inside Triton kernel — can't transfer to CPU during kernel execution",
            ))

        # Bug: tl.dot with wrong shapes (1D args)
        if "tl.dot(" in stripped:
            # Look for 1D tl.arange arguments being directly passed to tl.dot
            args_match = re.search(r"tl\.dot\(([^)]+)\)", stripped)
            if args_match:
                args = args_match.group(1)
                # Very rough check — can't fully parse without type info
                if "arange" in args:
                    result.issues.append(StaticIssue(
                        severity="warning",
                        category="correctness",
                        line=i,
                        message="tl.dot() with potentially 1D argument — tl.dot requires 2D inputs",
                        suggestion="Reshape with [:, None] or [None, :] to make 2D",
                    ))

        # Bug: hardcoded strides
        if re.search(r"\* (128|256|512|1024)\b", stripped) and "stride" not in stripped.lower():
            if "load" in stripped.lower() or "store" in stripped.lower() or "ptr" in stripped.lower():
                result.issues.append(StaticIssue(
                    severity="info",
                    category="correctness",
                    line=i,
                    message="Hardcoded stride value — may break for non-contiguous tensors",
                    suggestion="Use stride parameters from tensor metadata instead",
                ))


def _check_optimizations(source: str, result: StaticAnalysisResult, kernel_type: str):
    """Detect missed optimization opportunities."""

    # No autotune
    if "@triton.autotune" not in source and "triton.autotune" not in source:
        config_count = source.count("triton.Config")
        if config_count == 0:
            result.issues.append(StaticIssue(
                severity="info",
                category="performance",
                line=None,
                message="No @triton.autotune — add 5+ configs to explore tile sizes and num_warps",
            ))

    # No tl.dot (missed tensor cores)
    if "tl.dot" not in source:
        # Check if there are matrix-like operations
        has_matmul_pattern = bool(re.search(r"tl\.sum\([^)]*\*[^)]*axis", source))
        if has_matmul_pattern:
            result.issues.append(StaticIssue(
                severity="warning",
                category="performance",
                line=None,
                message="Matrix operations via tl.sum(a * b, axis=...) instead of tl.dot() — missing tensor cores (10-50x slower)",
                suggestion="Replace tl.sum(a[:, None, :] * b[None, :, :], axis=2) with tl.dot(a, b)",
            ))

    # Separate loops over same data
    loop_count = len(re.findall(r"for\s+\w+\s+in\s+range\(", source))
    load_count = source.count("tl.load")
    if loop_count > 2 and load_count > loop_count * 2:
        result.issues.append(StaticIssue(
            severity="info",
            category="performance",
            line=None,
            message=f"{loop_count} loops with {load_count} loads — possible unfused passes over same data",
            suggestion="Fuse operations into fewer loops to reduce memory traffic",
        ))

    # GDN-specific checks
    if kernel_type in ("gdn", "gated_delta_net"):
        # Check GVA head mapping
        if "num_v_heads" in source or "v_head" in source:
            if "//" not in source and "ratio" not in source.lower() and "group" not in source.lower():
                result.issues.append(StaticIssue(
                    severity="warning",
                    category="correctness",
                    line=None,
                    message="GVA head mapping likely missing — v_head to q/k head mapping via integer division",
                    suggestion="qk_head = v_head // (num_v_heads // num_q_heads)",
                ))

        # Check state precision
        if "state" in source.lower():
            state_loads = [i for i, l in enumerate(source.split("\n"), 1)
                          if "state" in l.lower() and "tl.load" in l]
            for line_num in state_loads:
                line = source.split("\n")[line_num - 1]
                if "bfloat16" in line or "float16" in line:
                    result.issues.append(StaticIssue(
                        severity="error",
                        category="correctness",
                        line=line_num,
                        message="State loaded as bf16/fp16 — GDN state MUST be float32",
                        suggestion="State is float32. Load and process in float32.",
                    ))


def _collect_metrics(source: str, result: StaticAnalysisResult):
    """Collect code complexity metrics."""
    lines = source.split("\n")
    result.metrics["total_lines"] = len(lines)
    result.metrics["code_lines"] = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    result.metrics["tl_load_count"] = source.count("tl.load")
    result.metrics["tl_store_count"] = source.count("tl.store")
    result.metrics["tl_dot_count"] = source.count("tl.dot")
    result.metrics["for_loops"] = len(re.findall(r"for\s+\w+\s+in\s+range\(", source))
    result.metrics["inline_asm"] = source.count("inline_asm_elementwise")

    # Count autotune configs
    configs = re.findall(r"triton\.Config", source)
    result.metrics["autotune_configs"] = len(configs)
