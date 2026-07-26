"""Load official SAM3 checkpoint weights into sam3 modules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sam3.weights.load_sam3 import (
    build_production_sam_head,
    build_production_vit,
    extract_sam_head_state_dict,
    extract_vit_trunk_state_dict,
    load_sam3_checkpoint,
    load_sam_head_weights,
    load_vit_trunk_weights,
    resolve_sam3_checkpoint,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required for real-weight tests", allow_module_level=True)

DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def ckpt_path() -> Path:
    try:
        return resolve_sam3_checkpoint()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def ckpt(ckpt_path: Path) -> dict:
    return load_sam3_checkpoint(str(ckpt_path))


def test_resolve_checkpoint(ckpt_path: Path):
    assert ckpt_path.is_file()
    assert ckpt_path.stat().st_size > 1_000_000


def test_extract_vit_and_sam_keys(ckpt: dict):
    trunk = extract_vit_trunk_state_dict(ckpt)
    head = extract_sam_head_state_dict(ckpt)
    assert any(k.startswith("blocks.0.attn.qkv") for k in trunk)
    assert any(k.endswith("freqs_cis_real") for k in trunk)
    assert any(k.startswith("prompt_encoder.") for k in head)
    assert any(k.startswith("mask_decoder.") for k in head)


def test_load_vit_trunk_weights(ckpt: dict):
    vit = build_production_vit()
    missing, skipped = load_vit_trunk_weights(vit, ckpt, strict=False)
    # Core weights must load; buffers like drop-path have none.
    loaded = len(vit.state_dict()) - len(missing)
    assert loaded > 100, f"too few keys loaded: {loaded}, missing sample {missing[:10]}"
    # qkv should match checkpoint exactly after load
    w = vit.blocks[0].attn.qkv.weight
    ref = ckpt["detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"]
    assert torch.equal(w.cpu(), ref)
    # freqs converted from complex
    fr = vit.blocks[0].attn.freqs_cis_real
    fc = ckpt["detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis"]
    assert torch.allclose(fr.cpu(), fc.real)


def test_load_sam_head_weights(ckpt: dict):
    head = build_production_sam_head()
    missing, skipped = load_sam_head_weights(head, ckpt, strict=False)
    loaded = len(head.state_dict()) - len(missing)
    assert loaded > 50, f"too few keys: {loaded}, missing={missing[:15]}"
    # prompt PE matrix
    pe = head.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix
    ref = ckpt[
        "tracker.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
    ]
    assert torch.equal(pe.cpu(), ref)
    # mask tokens
    assert torch.equal(
        head.mask_decoder.mask_tokens.weight.cpu(),
        ckpt["tracker.sam_mask_decoder.mask_tokens.weight"],
    )


@torch.inference_mode()
def test_sam_head_forward_with_real_weights(ckpt: dict):
    """Run tracker SAM head (256-d) with synthetic backbone features."""
    from sam3.dtype_policy import cast_module_

    head = build_production_sam_head()
    load_sam_head_weights(head, ckpt, strict=False)
    head = cast_module_(head, torch.float16, device=DEVICE).eval()

    b, c, h, w = 1, 256, 72, 72
    image_emb = torch.randn(b, c, h, w, device=DEVICE, dtype=torch.float16)
    # high-res feats for use_high_res_features path (same spatial hierarchy as SAM)
    # conv_s0/s1 expect 256-d FPN levels; provide matching channels
    high_res = [
        torch.randn(b, 256, h * 4, w * 4, device=DEVICE, dtype=torch.float16),
        torch.randn(b, 256, h * 2, w * 2, device=DEVICE, dtype=torch.float16),
    ]
    coords = torch.tensor([[[500.0, 500.0]]], device=DEVICE, dtype=torch.float16)
    labels = torch.tensor([[1]], device=DEVICE)

    # Bypass SamImageHead.forward high-res plumbing: call encoder + decoder
    sparse, dense = head.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    sparse = sparse.to(device=DEVICE, dtype=torch.float16)
    dense = dense.to(device=DEVICE, dtype=torch.float16)
    image_pe = head.prompt_encoder.get_dense_pe().to(device=DEVICE, dtype=torch.float16)

    # Process high-res like tracker: conv_s0/s1 on FPN
    feat_s0 = head.mask_decoder.conv_s0(high_res[0])
    feat_s1 = head.mask_decoder.conv_s1(high_res[1])

    masks, iou, tokens, obj = head.mask_decoder(
        image_embeddings=image_emb,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
        repeat_image=False,
        high_res_features=[feat_s0, feat_s1],
    )
    assert masks.ndim == 4 and masks.shape[0] == 1
    assert torch.isfinite(masks).all()
    assert torch.isfinite(iou).all()
    assert tokens.shape[-1] == 256


@torch.inference_mode()
def test_vit_trunk_partial_forward_with_real_weights(ckpt: dict):
    """Load full trunk; run a single window-block path on a small crop of features.

    Full 1008² × 32-layer forward is heavy; we verify weights + one block on GPU.
    """
    from sam3.dtype_policy import cast_module_

    vit = build_production_vit()
    load_vit_trunk_weights(vit, ckpt, strict=False)
    vit = cast_module_(vit, torch.float16, device=DEVICE).eval()

    # Use patch embed on a small random image is wrong size for rope; instead
    # feed NHWC tokens matching window size 24 into block 0.
    x = torch.randn(1, 24, 24, 1024, device=DEVICE, dtype=torch.float16)
    y = vit.blocks[0](x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
