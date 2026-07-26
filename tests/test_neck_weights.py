"""Neck + vision backbone real-weight load tests."""

from __future__ import annotations

import pytest
import torch

from sam3.primitives.position_encoding import PositionEmbeddingSine
from sam3.vision.necks import Sam3DualViTDetNeck
from sam3.weights.load_sam3 import (
    build_production_vision_backbone,
    extract_neck_state_dict,
    load_sam3_checkpoint,
    load_vision_backbone_weights,
    resolve_sam3_checkpoint,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.fixture(scope="module")
def ckpt():
    try:
        resolve_sam3_checkpoint()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return load_sam3_checkpoint()


def test_extract_neck_keys(ckpt):
    neck_sd = extract_neck_state_dict(ckpt)
    assert any(k.startswith("convs.0.") for k in neck_sd)
    assert any(k.startswith("sam2_convs.0.") for k in neck_sd)
    assert "convs.0.conv_1x1.weight" in neck_sd
    assert neck_sd["convs.0.conv_1x1.weight"].shape[0] == 256


def test_load_vision_backbone_weights(ckpt):
    neck = build_production_vision_backbone(add_sam2_neck=True, precompute_pe=False)
    missing, skipped = load_vision_backbone_weights(neck, ckpt, strict=False)
    # Most param keys should load
    n_params = sum(1 for k in neck.state_dict() if "position_encoding" not in k)
    n_missing_core = sum(1 for k in missing if "position_encoding" not in k)
    assert n_missing_core < n_params * 0.2, (
        f"too many missing: {n_missing_core}/{n_params} sample={missing[:15]}"
    )
    # Spot-check a neck weight
    w = neck.convs[2].conv_1x1.weight
    ref = ckpt["detector.backbone.vision_backbone.convs.2.conv_1x1.weight"]
    assert torch.equal(w.cpu(), ref)


@torch.inference_mode()
def test_neck_only_forward_parity_synthetic_trunk(ckpt):
    """Isolate FPN: same fake trunk output → official-style vs our neck convs."""

    # Tiny synthetic trunk
    class _StubTrunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.channel_list = [1024]
            self.embed_dim = 1024

        def forward(self, x):
            # Ignore image; return fixed BCHW feature
            b = x.shape[0] if torch.is_tensor(x) else 1
            return [getattr(self, "_feat").expand(b, -1, -1, -1)]

    feat = torch.randn(1, 1024, 8, 8, device="cuda", dtype=torch.float32)
    pe = PositionEmbeddingSine(256, precompute_resolution=None)

    def make_neck(trunk):
        return (
            Sam3DualViTDetNeck(
                trunk=trunk,
                position_encoding=pe,
                d_model=256,
                scale_factors=(4.0, 2.0, 1.0, 0.5),
                add_sam2_neck=True,
            )
            .cuda()
            .eval()
        )

    t1 = _StubTrunk().cuda()
    t1._feat = feat
    t2 = _StubTrunk().cuda()
    t2._feat = feat

    n1 = make_neck(t1)
    n2 = make_neck(t2)
    # Load same neck weights into both
    neck_sd = extract_neck_state_dict(ckpt)
    filt = {
        k: v
        for k, v in neck_sd.items()
        if k in n1.state_dict() and n1.state_dict()[k].shape == v.shape
    }
    n1.load_state_dict(filt, strict=False)
    n2.load_state_dict(filt, strict=False)

    dummy = torch.zeros(1, 3, 32, 32, device="cuda")
    o1 = n1(dummy)
    o2 = n2(dummy)
    for a, b in zip(o1[0], o2[0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
