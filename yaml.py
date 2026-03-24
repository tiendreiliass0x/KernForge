"""
Minimal YAML subset used by this repository's flat config files.

This is not a general YAML implementation. It only supports simple top-level
``key: value`` mappings with scalar values. The project still declares
``PyYAML`` as the canonical dependency for full environments.
"""

from __future__ import annotations

from collections.abc import Mapping


def _coerce_scalar(value: str):
    text = value.strip()
    if text == "":
        return ""

    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none", "~"}:
        return None

    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text


def safe_load(stream):
    if hasattr(stream, "read"):
        content = stream.read()
    else:
        content = str(stream)

    data = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, value = line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value)
    return data


def dump(data, stream=None, sort_keys=True):
    if not isinstance(data, Mapping):
        raise TypeError("yaml.dump only supports mappings in this project shim")

    items = sorted(data.items()) if sort_keys else data.items()
    lines = []
    for key, value in items:
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")

    output = "\n".join(lines) + ("\n" if lines else "")
    if stream is not None:
        stream.write(output)
        return None
    return output
