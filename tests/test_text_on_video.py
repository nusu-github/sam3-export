"""Smoke tests for the text-on-video runtime composition."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_associate_import() -> None:
    from sam3.runtime.associate_det_trk import associate_det_trk

    assert callable(associate_det_trk)


def test_build_text_on_video_shared_backbone() -> None:
    from sam3 import build_production_text_on_video

    model = build_production_text_on_video(load_weights=True)
    assert (
        model.detector.backbone.vision_backbone
        is model.tracker.backbone.vision_backbone
    )
    assert sum(parameter.numel() for parameter in model.parameters()) > 100_000_000
