"""CUDA export parity for internal B/C/F/G/H/I component fixtures."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.export import export

from sam3.export import (
    GroundingDecode,
    GroundingEncode,
    InteractiveImageEmbed,
    MemoryEncode,
    TextTower,
    TrackerStep,
)
from sam3.grounding.det_decoder import create_sam3_image_decoder
from sam3.grounding.det_encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.grounding.dot_product_scoring import DotProductScoring
from sam3.grounding.seg_head import PixelDecoder, UniversalSegmentationHead
from sam3.grounding.text_encoder_ve import VETextEncoder
from sam3.grounding.tokenizer_ve import SimpleTokenizer
from sam3.primitives.mlp import MLP
from sam3.tracking.sam3_tracker import _build_maskmem_backbone, build_tiny_sam3_tracker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

DEVICE = torch.device("cuda")


def _roundtrip(module: nn.Module, args: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = module.to(DEVICE).eval()
    with torch.no_grad():
        eager = module(*args)
        exported = export(module, args, strict=False).module()(*args)
    assert isinstance(eager, tuple) and isinstance(exported, tuple)
    assert len(eager) == len(exported)
    for expected, actual in zip(eager, exported):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(expected, actual, rtol=1e-3, atol=1e-3)
    return exported


def _tiny_grounding_encoder(d_model: int = 32) -> TransformerEncoderFusion:
    layer = TransformerEncoderLayer(
        activation="relu",
        cross_attention=nn.MultiheadAttention(d_model, 4, batch_first=True),
        d_model=d_model,
        dim_feedforward=64,
        dropout=0.0,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=nn.MultiheadAttention(d_model, 4, batch_first=True),
    )
    return TransformerEncoderFusion(
        layer=layer,
        num_layers=1,
        d_model=d_model,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=False,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


@torch.no_grad()
def test_text_tower_ids_only_export() -> None:
    tower = TextTower(
        VETextEncoder(
            d_model=32,
            tokenizer=SimpleTokenizer(),
            width=64,
            heads=4,
            layers=1,
            context_length=8,
            use_act_checkpoint=False,
        ),
        validate=False,
    )
    # TextTransformer intentionally leaves checkpoint-owned parameters empty;
    # initialize this synthetic smoke instance before asserting finite parity.
    for parameter in tower.parameters():
        nn.init.normal_(parameter, std=0.02)
    ids = torch.tensor([[49406, 42, 49407, 0, 0, 0, 0, 0]], device=DEVICE)
    memory, padding = _roundtrip(tower, (ids, ids.ne(0)))
    assert memory.shape == (1, 8, 32)
    assert padding.tolist() == [[False, False, False, True, True, True, True, True]]


@torch.no_grad()
def test_interactive_image_embed_and_memory_encode_export() -> None:
    tracker = build_tiny_sam3_tracker().to(DEVICE).eval()
    view = InteractiveImageEmbed(tracker)
    fpn = (
        torch.randn(1, 64, 32, 32, device=DEVICE),
        torch.randn(1, 64, 16, 16, device=DEVICE),
        torch.randn(1, 64, 8, 8, device=DEVICE),
        torch.randn(1, 64, 4, 4, device=DEVICE),
    )
    image_embed, high_res_0, high_res_1 = _roundtrip(view, fpn)
    assert image_embed.shape == (1, 64, 8, 8)
    assert high_res_0.shape == (1, 8, 32, 32)
    assert high_res_1.shape == (1, 16, 16, 16)

    mem_encoder = _build_maskmem_backbone(64, 16, image_size=64, backbone_stride=8)
    memory_cut = MemoryEncode(mem_encoder)
    memory, memory_pos = _roundtrip(
        memory_cut,
        (image_embed, torch.randn(1, 1, 64, 64, device=DEVICE)),
    )
    assert memory.shape == memory_pos.shape == (1, 16, 8, 8)


@torch.no_grad()
def test_grounding_encode_decode_export() -> None:
    d_model, batch, text_len, hw = 32, 2, 5, 4
    encoder_cut = GroundingEncode(_tiny_grounding_encoder(d_model), 1)
    features = torch.randn(batch, d_model, hw, hw, device=DEVICE)
    positions = torch.randn_like(features)
    image_mask = torch.zeros(batch, hw, hw, dtype=torch.bool, device=DEVICE)
    text_memory = torch.randn(batch, text_len, d_model, device=DEVICE)
    text_mask = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, True]],
        device=DEVICE,
    )
    encoded = _roundtrip(
        encoder_cut,
        ((features,), (positions,), (image_mask,), text_memory, text_mask),
    )
    memory, pos, padding, starts, shapes, ratios, text_after = encoded
    assert memory.shape == pos.shape == (hw * hw, batch, d_model)
    assert padding.shape == (batch, hw * hw)
    assert text_after.shape == text_memory.shape

    decoder = create_sam3_image_decoder(
        d_model=d_model,
        n_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        num_layers=1,
        num_queries=3,
        resolution=32,
        stride=8,
    )
    scorer = DotProductScoring(
        d_model,
        d_model,
        MLP(
            d_model,
            64,
            d_model,
            2,
            dropout=0.0,
            residual=True,
            out_norm=nn.LayerNorm(d_model),
        ),
    )
    segmentation = UniversalSegmentationHead(
        hidden_dim=d_model,
        upsampling_stages=1,
        pixel_decoder=PixelDecoder(d_model, 1),
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=False,
        cross_attend_prompt=None,
    )
    decode_cut = GroundingDecode(decoder, scorer, segmentation)
    logits, boxes, masks, presence = _roundtrip(
        decode_cut,
        (
            (features,),
            memory,
            pos,
            padding,
            starts,
            shapes,
            ratios,
            text_after,
            text_mask,
        ),
    )
    assert logits.shape == (batch, 3, 1)
    assert boxes.shape == (batch, 3, 4)
    assert masks.shape == (batch, 3, hw, hw)
    assert presence.shape == (batch, 1)


@torch.no_grad()
def test_tracker_step_padded_memory_export() -> None:
    tracker = build_tiny_sam3_tracker().to(DEVICE).eval()
    cut = TrackerStep(tracker)
    batch, hw, memory_slots, pointers = 1, 8, 2, 1
    args = (
        torch.randn(batch, 64, hw, hw, device=DEVICE),
        torch.randn(batch, 64, hw, hw, device=DEVICE),
        torch.randn(batch, 8, 32, 32, device=DEVICE),
        torch.randn(batch, 16, 16, 16, device=DEVICE),
        torch.randn(batch, memory_slots, 16, hw, hw, device=DEVICE),
        torch.randn(batch, memory_slots, 16, hw, hw, device=DEVICE),
        torch.tensor([[False, True]], device=DEVICE),
        torch.randn(batch, pointers, 16, device=DEVICE),
        torch.randn(batch, pointers, 16, device=DEVICE),
        torch.zeros(batch, pointers, dtype=torch.bool, device=DEVICE),
        torch.tensor([[[10.0, 10.0]]], device=DEVICE),
        torch.ones(batch, 1, dtype=torch.long, device=DEVICE),
    )
    low_res, high_res, obj_ptr, obj_score = _roundtrip(cut, args)
    assert low_res.shape == (batch, 1, 32, 32)
    assert high_res.shape == (batch, 1, 64, 64)
    assert obj_ptr.shape == (batch, 64)
    assert obj_score.shape == (batch, 1)

    # The second spatial slot is padded. Its values must not affect the step.
    changed = list(args)
    changed_memory = args[4].clone()
    changed_memory[:, 1].fill_(10_000.0)
    changed[4] = changed_memory
    with torch.no_grad():
        changed_out = cut(*changed)
    for expected, actual in zip((low_res, high_res, obj_ptr, obj_score), changed_out):
        torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-4)
