"""Shared M6 helpers for serializing canonical ``ExportedProgram`` captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _argument_name(argument: object) -> str | None:
    name = getattr(argument, "name", None)
    return str(name) if name is not None else None


def _signature_spec(spec: object) -> dict[str, object]:
    kind = getattr(spec, "kind")
    kind_name = getattr(kind, "name", str(kind))
    target = getattr(spec, "target", None)
    persistent = getattr(spec, "persistent", None)
    record: dict[str, object] = {
        "kind": str(kind_name).lower().replace("_", "-"),
        "argument_type": type(getattr(spec, "arg")).__name__,
        "argument_name": _argument_name(getattr(spec, "arg")),
        "target": None if target is None else str(target),
    }
    if persistent is not None:
        record["persistent"] = bool(persistent)
    return record


def save_exported_program(
    program: torch.export.ExportedProgram,
    path: Path,
    *,
    bundle_path: str,
    input_names: list[str],
    output_names: list[str],
    mode: str,
) -> dict[str, Any]:
    """Save one canonical capture and return its manifest-owned metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, path)
    signature = program.graph_signature
    return {
        "program_path": bundle_path,
        "capture_mode": mode,
        "semantic_inputs": input_names,
        "semantic_outputs": output_names,
        "graph_signature": {
            "inputs": [_signature_spec(spec) for spec in signature.input_specs],
            "outputs": [_signature_spec(spec) for spec in signature.output_specs],
        },
        "range_constraints": [
            {"symbol": str(symbol), "range": str(value)}
            for symbol, value in sorted(
                program.range_constraints.items(), key=lambda item: str(item[0])
            )
        ],
    }


__all__ = ["save_exported_program"]
