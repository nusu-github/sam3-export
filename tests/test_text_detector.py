"""Integration tests for the text open-vocab detector path."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_build_text_detector_shapes():
    from sam3 import build_production_text_detector

    model = build_production_text_detector(enable_segmentation=True)
    assert model.transformer.d_model == 256
    assert model.transformer.decoder.num_queries == 200
    assert model.segmentation_head is not None
    n = sum(p.numel() for p in model.parameters())
    assert n > 100_000_000  # production-scale


def test_load_text_detector_weights():
    from sam3 import (
        build_production_text_detector,
        load_text_detector_weights,
        resolve_sam3_checkpoint,
    )

    resolve_sam3_checkpoint()  # raises if missing
    model = build_production_text_detector()
    missing, _skipped = load_text_detector_weights(model)
    # Most detector keys should load; freqs real/imag may leave some complex skips
    # Missing should not include critical roots
    critical_missing = [
        m
        for m in missing
        if any(
            m.startswith(p)
            for p in (
                "backbone.language_backbone.encoder.token_embedding",
                "transformer.decoder.query_embed",
                "transformer.encoder.layers.0",
                "dot_prod_scoring.hs_proj",
            )
        )
    ]
    assert not critical_missing, critical_missing[:10]
    # Expect high load rate
    assert len(missing) < 50, (
        f"too many missing keys: {len(missing)} e.g. {missing[:15]}"
    )


@torch.inference_mode()
def test_text_forward_smoke_tiny_image():
    """Forward on a tiny random image after weight load (may be slow once)."""
    from PIL import Image

    from sam3 import Sam3TextPredictor, build_production_text_detector

    model = build_production_text_detector(
        dtype=torch.bfloat16,
        device="cuda",
        load_weights=True,
    )
    assert next(model.parameters()).dtype == torch.bfloat16
    pred = Sam3TextPredictor(model=model, device="cuda", confidence_threshold=0.01)
    # Use small synthetic image — still resized to 1008 internally
    img = Image.new("RGB", (256, 256), color=(128, 90, 40))
    state = pred.set_image(img)
    state = pred.set_text_prompt("object", state)
    assert "scores" in state and "masks" in state and "boxes" in state
    assert state["masks"].ndim >= 3 or state["masks"].numel() == 0
