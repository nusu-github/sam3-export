"""Real SAM3.1 checkpoint identity and target-shape gates."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sam3.weights.multiplex import (
    SAM31_CHECKPOINT_SHA256,
    SAM31_REVISION,
    TRI_NECK_PREFIX,
    build_sam31_multiplex_tracker_core,
    build_sam31_tri_neck,
    load_sam31_multiplex_checkpoint,
    map_checkpoint_to_module,
    resolve_sam31_multiplex_checkpoint,
    verify_multiplex_checkpoint_shapes,
)


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    try:
        return resolve_sam31_multiplex_checkpoint()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_checkpoint_revision_digest_and_owned_shapes(checkpoint_path: Path) -> None:
    checkpoint = load_sam31_multiplex_checkpoint(checkpoint_path)
    assert len(checkpoint) == 1623
    verify_multiplex_checkpoint_shapes(checkpoint)


def test_local_tri_neck_mapping_is_name_shape_and_value_exact(
    checkpoint_path: Path,
) -> None:
    checkpoint = load_sam31_multiplex_checkpoint(checkpoint_path)
    module = build_sam31_tri_neck(checkpoint)
    report = map_checkpoint_to_module(
        checkpoint, module, prefix=TRI_NECK_PREFIX, load=False
    )
    assert report.exact
    # 32 complex RoPE buffers are represented as 64 real/imag buffers locally.
    assert report.checkpoint_key_count == report.module_key_count == 506


def test_local_tracker_mapping_has_no_missing_or_unexpected_target(
    checkpoint_path: Path,
) -> None:
    checkpoint = load_sam31_multiplex_checkpoint(checkpoint_path)
    module = build_sam31_multiplex_tracker_core(checkpoint)
    report = map_checkpoint_to_module(
        checkpoint, module, prefix="tracker.model.", load=False
    )
    assert report.exact
    assert report.checkpoint_key_count == report.module_key_count == 457


def test_checkpoint_revision_mismatch_is_explicit(checkpoint_path: Path) -> None:
    with pytest.raises(ValueError, match="revision mismatch"):
        load_sam31_multiplex_checkpoint(checkpoint_path, revision="wrong")


def test_checkpoint_digest_mismatch_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    torch.save({"parameter": torch.ones(1)}, checkpoint)
    monkeypatch.setattr("sam3.weights.multiplex.SAM31_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_sam31_multiplex_checkpoint(checkpoint)


def test_fixed_identity_constants_are_not_ambiguous() -> None:
    assert SAM31_REVISION == "daa63191845a41281374e725f4c9e51c7a824460"
    assert SAM31_CHECKPOINT_SHA256 == (
        "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
    )
