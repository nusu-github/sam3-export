"""CUDA unit tests for segmentation head + DotProductScoring."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sam3.grounding.dot_product_scoring import DotProductScoring
from sam3.grounding.seg_head import (
    MaskPredictor,
    PixelDecoder,
    UniversalSegmentationHead,
)
from sam3.primitives.mlp import MLP

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for seg_head tests", allow_module_level=True)

DEVICE = torch.device("cuda")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# PixelDecoder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_dim,stages,fine_to_coarse",
    [
        # fine → coarse (matches SAM3 backbone neck order)
        (32, 3, [(64, 64), (32, 32), (16, 16), (8, 8)]),
        (16, 2, [(16, 16), (8, 8), (4, 4)]),
    ],
)
def test_pixel_decoder_shapes(
    hidden_dim: int,
    stages: int,
    fine_to_coarse: list[tuple[int, int]],
) -> None:
    """FPN top-down: starts at coarsest, ends at finest spatial size."""
    torch.manual_seed(0)
    B = 2
    feats = [
        torch.randn(B, hidden_dim, h, w, device=DEVICE, dtype=DTYPE)
        for h, w in fine_to_coarse
    ]

    dec = PixelDecoder(
        hidden_dim=hidden_dim,
        num_upsampling_stages=stages,
        interpolation_mode="nearest",
    ).to(DEVICE)
    dec.eval()

    with torch.no_grad():
        out = dec(feats)

    finest_h, finest_w = fine_to_coarse[0]
    assert out.shape == (B, hidden_dim, finest_h, finest_w)
    assert torch.isfinite(out).all()
    assert dec.out_dim == hidden_dim


def test_pixel_decoder_upsample_8_16_32_64() -> None:
    """Explicit production-like relative scales: 8×8 … 64×64."""
    torch.manual_seed(1)
    hidden = 32
    B = 1
    # fine → coarse order (matches backbone neck order used by SAM3)
    sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    feats = [torch.randn(B, hidden, h, w, device=DEVICE, dtype=DTYPE) for h, w in sizes]
    # num_upsampling_stages = len(feats) - 1 = 3
    dec = PixelDecoder(hidden_dim=hidden, num_upsampling_stages=3).to(DEVICE)
    dec.eval()
    with torch.no_grad():
        out = dec(feats)
    assert out.shape == (B, hidden, 64, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MaskPredictor
# ---------------------------------------------------------------------------


def test_mask_predictor_einsum_shapes() -> None:
    torch.manual_seed(2)
    hidden = 32
    B, Q, H, W = 2, 5, 16, 16
    L = 3

    pred = MaskPredictor(hidden_dim=hidden, mask_dim=hidden).to(DEVICE)
    pred.eval()

    obj_q = torch.randn(B, Q, hidden, device=DEVICE, dtype=DTYPE)
    pix = torch.randn(B, hidden, H, W, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        masks = pred(obj_q, pix)
    assert masks.shape == (B, Q, H, W)
    assert torch.isfinite(masks).all()

    # batch omitted on pixel embed
    pix_nb = torch.randn(hidden, H, W, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        masks_nb = pred(obj_q, pix_nb)
    assert masks_nb.shape == (B, Q, H, W)

    # aux layers
    obj_aux = torch.randn(L, B, Q, hidden, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        masks_aux = pred(obj_aux, pix)
    assert masks_aux.shape == (L, B, Q, H, W)


# ---------------------------------------------------------------------------
# DotProductScoring
# ---------------------------------------------------------------------------


def test_dot_product_scoring_forward() -> None:
    torch.manual_seed(3)
    d_model, d_proj = 32, 16
    L, B, Q, S = 2, 2, 4, 6

    prompt_mlp = MLP(
        input_dim=d_model,
        hidden_dim=64,
        output_dim=d_model,
        num_layers=2,
        residual=True,
        out_norm=nn.LayerNorm(d_model),
    )
    scorer = DotProductScoring(
        d_model=d_model, d_proj=d_proj, prompt_mlp=prompt_mlp
    ).to(DEVICE)
    scorer.eval()

    hs = torch.randn(L, B, Q, d_model, device=DEVICE, dtype=DTYPE)
    prompt = torch.randn(S, B, d_model, device=DEVICE, dtype=DTYPE)
    prompt_mask = torch.zeros(B, S, device=DEVICE, dtype=torch.bool)
    prompt_mask[:, -2:] = True  # last 2 tokens padded

    with torch.no_grad():
        scores = scorer(hs, prompt, prompt_mask)

    assert scores.shape == (L, B, Q, 1)
    assert torch.isfinite(scores).all()
    assert scores.abs().max() <= 12.0


# ---------------------------------------------------------------------------
# UniversalSegmentationHead
# ---------------------------------------------------------------------------


def test_universal_segmentation_head_forward_keys() -> None:
    """presence_head=False → keys pred_masks, semantic_seg, presence_logit=None."""
    torch.manual_seed(5)
    hidden = 32
    B, Q, S = 2, 4, 6
    # fine → coarse
    sizes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    backbone_feats = [
        torch.randn(B, hidden, h, w, device=DEVICE, dtype=DTYPE) for h, w in sizes
    ]
    # encoder memory sequence length = H*W of coarsest = 4*4
    spatial = sizes[-1][0] * sizes[-1][1]
    encoder_hs = torch.randn(spatial, B, hidden, device=DEVICE, dtype=DTYPE)
    # obj_queries: (num_layers, B, Q, C) — last layer used when aux_masks=False
    obj_queries = torch.randn(2, B, Q, hidden, device=DEVICE, dtype=DTYPE)
    image_ids = torch.arange(B, device=DEVICE)
    prompt = torch.randn(S, B, hidden, device=DEVICE, dtype=DTYPE)
    prompt_mask = torch.zeros(B, S, device=DEVICE, dtype=torch.bool)

    pixel_decoder = PixelDecoder(hidden_dim=hidden, num_upsampling_stages=3).to(DEVICE)

    cross_attn = nn.MultiheadAttention(
        embed_dim=hidden, num_heads=4, dropout=0.0, batch_first=False
    ).to(DEVICE)

    head = UniversalSegmentationHead(
        hidden_dim=hidden,
        upsampling_stages=3,
        pixel_decoder=pixel_decoder,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=False,
        cross_attend_prompt=cross_attn,
    ).to(DEVICE)
    head.eval()

    with torch.no_grad():
        out = head(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hs,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

    assert set(out.keys()) == {"pred_masks", "semantic_seg", "presence_logit"}
    assert out["presence_logit"] is None
    # After FPN, spatial is finest (32×32); pixel embeds are per image_ids
    pred_masks = out["pred_masks"]
    semantic = out["semantic_seg"]
    assert pred_masks.ndim == 4  # (B, Q, H, W) after indexing
    assert pred_masks.shape[0] == B
    assert pred_masks.shape[1] == Q
    assert pred_masks.shape[-2:] == sizes[0]
    assert semantic.shape[0] == B
    assert semantic.shape[1] == 1
    assert semantic.shape[-2:] == sizes[0]
    assert torch.isfinite(pred_masks).all()
    assert torch.isfinite(semantic).all()


def test_universal_segmentation_head_no_cross_attn() -> None:
    """cross_attend_prompt=None still produces the three output keys."""
    torch.manual_seed(6)
    hidden = 16
    B, Q = 1, 3
    sizes = [(16, 16), (8, 8), (4, 4)]
    backbone_feats = [
        torch.randn(B, hidden, h, w, device=DEVICE, dtype=DTYPE) for h, w in sizes
    ]
    spatial = sizes[-1][0] * sizes[-1][1]
    encoder_hs = torch.randn(spatial, B, hidden, device=DEVICE, dtype=DTYPE)
    obj_queries = torch.randn(1, B, Q, hidden, device=DEVICE, dtype=DTYPE)
    image_ids = torch.zeros(B, device=DEVICE, dtype=torch.long)

    pixel_decoder = PixelDecoder(hidden_dim=hidden, num_upsampling_stages=2).to(DEVICE)
    head = UniversalSegmentationHead(
        hidden_dim=hidden,
        upsampling_stages=2,
        pixel_decoder=pixel_decoder,
        presence_head=False,
        cross_attend_prompt=None,
        act_ckpt=False,
    ).to(DEVICE)
    head.eval()

    with torch.no_grad():
        out = head(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hs,
        )
    assert "pred_masks" in out and "semantic_seg" in out
    assert out["presence_logit"] is None
    # use_encoder_inputs=True keeps batch dim; B=1 → (1, Q, H, W)
    assert out["pred_masks"].shape == (B, Q, *sizes[0])
    assert torch.isfinite(out["pred_masks"]).all()
