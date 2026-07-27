"""M6 gates for saved canonical ``ExportedProgram`` metadata."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from capture_utils import save_exported_program  # noqa: E402


class _Scale(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * 2


def test_saved_exported_program_round_trip_and_metadata(tmp_path: Path) -> None:
    values = torch.ones(2, 3)
    batch = torch.export.Dim("batch", min=1, max=4)
    program = torch.export.export(
        _Scale(),
        (values,),
        dynamic_shapes={"values": {0: batch}},
        strict=False,
    )
    path = tmp_path / "capture/scale.pt2"

    metadata = save_exported_program(
        program,
        path,
        bundle_path="capture/scale.pt2",
        input_names=["values"],
        output_names=["scaled-values"],
        mode="non-strict",
    )

    loaded = torch.export.load(path)
    torch.testing.assert_close(loaded.module()(values), values * 2)
    assert metadata["program_path"] == "capture/scale.pt2"
    assert metadata["semantic_inputs"] == ["values"]
    assert metadata["semantic_outputs"] == ["scaled-values"]
    assert metadata["range_constraints"]
    assert metadata["graph_signature"]["inputs"][-1]["kind"] == "user-input"
