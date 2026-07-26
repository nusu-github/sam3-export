"""Real-weight interactive mask path (neck → SAM head)."""

from __future__ import annotations

from PIL import Image
import pytest
import torch

from sam3.weights.load_sam3 import (
    build_production_interactive,
    resolve_sam3_checkpoint,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.fixture(
    params=[torch.bfloat16, torch.float16],
    scope="module",
    ids=["bf16", "fp16"],
)
def dtype(request):
    return request.param


@pytest.fixture(
    scope="module", params=[torch.bfloat16, torch.float16], ids=["bf16", "fp16"]
)
def predictor(dtype):
    try:
        resolve_sam3_checkpoint()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    pred = build_production_interactive(
        dtype=dtype,
        device="cuda",
        load_weights=True,
    )
    # Spot-check permanent cast landed on floating weights
    w = next(p for p in pred.parameters() if p.is_floating_point())
    assert w.dtype == dtype
    pred.eval()
    return pred


@torch.inference_mode()
def test_interactive_shapes_and_scores(predictor, dtype):
    image = Image.new("RGB", (1800, 1200), color=(128, 90, 40))
    orig_hw = predictor.set_image(image, dtype=dtype)
    assert orig_hw == (1200, 1800)

    masks, scores, low_res = predictor.predict(
        point_coords=[[500.0, 375.0]],
        point_labels=[1],
        multimask_output=True,
    )
    assert masks.shape[0] == 3
    assert masks.shape[-2:] == orig_hw
    assert scores.shape == (3,)
    assert low_res.shape[0] == 3
    assert torch.isfinite(scores).all()
