"""Load official SAM3 checkpoints into sam3 modules.

SAM3 ``sam3.pt`` layout (high level)::

    detector.backbone.vision_backbone.trunk.*   → ViT image encoder
    detector.backbone.language_backbone.*       → text encoder
    detector.transformer.*                      → DETR enc/dec
    detector.geometry_encoder.* / segmentation_head / dot_prod_scoring
    tracker.sam_prompt_encoder.*                → PromptEncoder
    tracker.sam_mask_decoder.*                  → MaskDecoder + TwoWayTransformer

The detector ViT is 1024-d / depth-32; the tracker SAM head is 256-d.
They are **not** the same tensor width — use the builders below separately,
or put a neck between them for a full cascade.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from sam3.dtype_policy import (
    PrecisionConfig,
    finalize_module,
    resolve_device,
    resolve_precision,
)
from sam3.grounding.det_decoder import create_sam3_image_decoder
from sam3.grounding.det_encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
)
from sam3.grounding.dot_product_scoring import DotProductScoring
from sam3.grounding.geometry_encoders import SequenceGeometryEncoder
from sam3.grounding.sam3_image import Sam3Image
from sam3.grounding.sam3_text_predictor import Sam3TextPredictor
from sam3.grounding.seg_head import PixelDecoder, UniversalSegmentationHead
from sam3.grounding.text_encoder_ve import VETextEncoder
from sam3.grounding.tokenizer_ve import DEFAULT_BPE_PATH, SimpleTokenizer
from sam3.grounding.transformer_wrapper import TransformerWrapper
from sam3.grounding.vl_combiner import SAM3VLBackbone
from sam3.primitives.mlp import MLP
from sam3.primitives.position_encoding import PositionEmbeddingSine
from sam3.tracking.sam3_text_video import Sam3TextOnVideo
from sam3.tracking.sam3_tracker import Sam3Tracker, build_sam3_tracker
from sam3.tracking.sam3_video_tracker import Sam3VideoTracker
from sam3.vision.necks import Sam3DualViTDetNeck
from sam3.vision.sam_image_head import SamImageHead
from sam3.vision.sam_interactive import SamInteractivePredictor
from sam3.vision.vit import ViT

SAM3_HF_REPO = "facebook/sam3"
DEFAULT_SAM3_CHECKPOINT = (
    "/workspace/.cache/huggingface/hub/models--facebook--sam3/"
    "snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
)
_VIT_PREFIX = "detector.backbone.vision_backbone.trunk."
_NECK_PREFIX = "detector.backbone.vision_backbone."
_PROMPT_PREFIX = "tracker.sam_prompt_encoder."
_MASK_PREFIX = "tracker.sam_mask_decoder."
_DETECTOR_PREFIX = "detector."
_TRACKER_PREFIX = "tracker."


def resolve_sam3_checkpoint(path: Optional[str] = None) -> Path:
    """Resolve a local checkpoint path (default: cached HF sam3.pt)."""
    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("SAM3_CHECKPOINT")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(DEFAULT_SAM3_CHECKPOINT))
    # Common HF cache patterns
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates.extend(hub.glob("hub/models--facebook--sam3/snapshots/*/sam3.pt"))

    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "SAM3 checkpoint not found. Pass path=..., set SAM3_CHECKPOINT, "
        f"or download facebook/sam3 (sam3.pt). Tried: {candidates[:4]}..."
    )


def load_sam3_checkpoint(path: Optional[str] = None) -> dict[str, torch.Tensor]:
    """Load raw flat state dict from ``sam3.pt``."""
    ckpt_path = resolve_sam3_checkpoint(path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(raw)}")
    return raw  # type: ignore[return-value]


def extract_vit_trunk_state_dict(
    ckpt: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map ``detector...trunk.*`` → ViT module keys (with freqs_cis → real/imag)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if not k.startswith(_VIT_PREFIX):
            continue
        name = k[len(_VIT_PREFIX) :]
        if name.endswith("attn.freqs_cis") and torch.is_complex(v):
            base = name[: -len("freqs_cis")]
            out[base + "freqs_cis_real"] = v.real.contiguous()
            out[base + "freqs_cis_imag"] = v.imag.contiguous()
            continue
        out[name] = v
    if not out:
        raise KeyError("No ViT trunk keys found under " + _VIT_PREFIX)
    return out


def extract_sam_head_state_dict(
    ckpt: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map tracker SAM prompt/mask decoder keys → ``SamImageHead`` keys."""
    out: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if k.startswith(_PROMPT_PREFIX):
            out["prompt_encoder." + k[len(_PROMPT_PREFIX) :]] = v
        elif k.startswith(_MASK_PREFIX):
            # Mask decoder owns the TwoWayTransformer as ``transformer``.
            rest = k[len(_MASK_PREFIX) :]
            out["mask_decoder." + rest] = v
            # Our SamImageHead also holds ``self.transformer`` alias of mask_decoder.transformer
            if rest.startswith("transformer."):
                out[rest] = v
    if not out:
        raise KeyError("No SAM head keys found under tracker.sam_*")
    return out


def build_production_vit(**overrides: Any) -> ViT:
    """ViT matching SAM3 ``_create_vit_backbone`` (model_builder.py)."""
    cfg = dict(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        # Production sets rope_pt_size to window when None in Block; keep explicit.
        rope_pt_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        bias_patch_embed=False,
        use_rope_real=True,  # sam3 stores real/imag buffers
        use_rel_pos_blocks=False,
        drop_path_rate=0.0,  # inference; ckpt has no drop-path params
        dropout=0.0,
    )
    cfg.update(overrides)
    return ViT(**cfg)


def build_production_sam_head(
    **overrides: Any,
) -> SamImageHead:
    """SAM-style head matching tracker ``_build_sam_heads`` (256-d, depth 2)."""
    embed_dim = int(overrides.pop("embed_dim", 256))
    image_size = int(overrides.pop("image_size", 1008))
    backbone_stride = int(overrides.pop("backbone_stride", 14))
    grid = image_size // backbone_stride  # 72
    cfg = dict(
        embed_dim=embed_dim,
        image_embedding_size=(grid, grid),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
        transformer_depth=2,
        transformer_heads=8,
        transformer_mlp_dim=2048,
        num_multimask_outputs=3,
        use_high_res_features=True,
        iou_prediction_use_sigmoid=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        iou_head_hidden_dim=256,
    )
    cfg.update(overrides)
    return SamImageHead(**cfg)


def _filter_state_dict_for_module(
    module: nn.Module, state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Keep only keys present in ``module`` with matching shapes."""
    model_sd = module.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            continue
        filtered[k] = v
    return filtered


def load_vit_trunk_weights(
    vit: ViT,
    ckpt: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    checkpoint_path: Optional[str] = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load detector ViT trunk weights into a sam3 ``ViT``.

    Returns ``(missing_keys, unexpected_or_skipped)``.
    """
    if ckpt is None:
        ckpt = load_sam3_checkpoint(checkpoint_path)
    trunk = extract_vit_trunk_state_dict(ckpt)
    filtered = _filter_state_dict_for_module(vit, trunk)
    missing, unexpected = vit.load_state_dict(filtered, strict=False)
    skipped = [k for k in trunk if k not in filtered]
    if strict and (missing or skipped):
        raise RuntimeError(f"strict load failed: {missing=} {skipped[:10]=}")
    return list(missing), skipped + list(unexpected)


def load_sam_head_weights(
    head: SamImageHead,
    ckpt: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    checkpoint_path: Optional[str] = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load tracker SAM prompt encoder + mask decoder into ``SamImageHead``."""
    if ckpt is None:
        ckpt = load_sam3_checkpoint(checkpoint_path)
    head_sd = extract_sam_head_state_dict(ckpt)
    # Prefer nested mask_decoder.* keys; drop duplicate bare transformer.*
    prefer = {
        k: v
        for k, v in head_sd.items()
        if k.startswith("prompt_encoder.") or k.startswith("mask_decoder.")
    }
    filtered = _filter_state_dict_for_module(head, prefer)
    missing, unexpected = head.load_state_dict(filtered, strict=False)
    skipped = [k for k in prefer if k not in filtered]
    if strict and (missing or skipped):
        raise RuntimeError(f"strict load failed: {missing=} {skipped[:10]=}")
    return list(missing), skipped + list(unexpected)


def extract_neck_state_dict(
    ckpt: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map ``detector...vision_backbone.convs/sam2_convs.*`` → neck keys."""
    out: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if not k.startswith(_NECK_PREFIX):
            continue
        rest = k[len(_NECK_PREFIX) :]
        if rest.startswith("trunk."):
            continue  # trunk handled separately
        # rest is like "convs.0.conv_1x1.weight"
        out[rest] = v
    if not out:
        raise KeyError("No neck conv keys found under " + _NECK_PREFIX)
    return out


def build_production_vision_backbone(
    *,
    add_sam2_neck: bool = True,
    precompute_pe: bool = False,
    precision: PrecisionConfig | None = None,
    dtype: torch.dtype | str | None = None,
    device: str | torch.device | None = None,
    load_weights: bool = False,
    checkpoint_path: Optional[str] = None,
    use_autocast: bool = False,
    **vit_overrides: Any,
) -> Sam3DualViTDetNeck:
    """Production ViT trunk + DualViTDet neck (d_model=256).

    When ``load_weights=True`` without an explicit dtype, defaults to
    :data:`~sam3.dtype_policy.DEFAULT_INFERENCE_DTYPE` permanent cast.
    """
    precision = resolve_precision(
        precision,
        dtype=dtype,
        use_autocast=use_autocast,
        load_weights=load_weights,
    )
    device = resolve_device(device, precision=precision)
    vit = build_production_vit(**vit_overrides)
    pe = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008 if precompute_pe else None,
    )
    neck = Sam3DualViTDetNeck(
        trunk=vit,
        position_encoding=pe,
        d_model=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=add_sam2_neck,
    )
    if load_weights:
        load_vision_backbone_weights(neck, checkpoint_path=checkpoint_path)
    finalize_module(neck, precision, device=device)
    neck.eval()
    return neck


def load_vision_backbone_weights(
    neck: Sam3DualViTDetNeck,
    ckpt: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    checkpoint_path: Optional[str] = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load trunk + FPN neck weights into ``Sam3DualViTDetNeck``."""
    if ckpt is None:
        ckpt = load_sam3_checkpoint(checkpoint_path)

    # Trunk
    trunk_sd = extract_vit_trunk_state_dict(ckpt)
    trunk_filtered = {f"trunk.{k}": v for k, v in trunk_sd.items()}
    # Neck convs
    neck_sd = extract_neck_state_dict(ckpt)
    combined = {**trunk_filtered, **neck_sd}
    filtered = _filter_state_dict_for_module(neck, combined)
    missing, unexpected = neck.load_state_dict(filtered, strict=False)
    skipped = [k for k in combined if k not in filtered]
    if strict and (missing or skipped):
        raise RuntimeError(f"strict load failed: {missing=} {skipped[:10]=}")
    return list(missing), skipped + list(unexpected)


def build_production_interactive(
    *,
    precompute_pe: bool = False,
    precision: PrecisionConfig | None = None,
    dtype: torch.dtype | str | None = None,
    device: str | torch.device | None = None,
    load_weights: bool = False,
    checkpoint_path: Optional[str] = None,
    use_autocast: bool = False,
    **vit_overrides: Any,
) -> SamInteractivePredictor:
    """Full interactive stack: Dual neck (sam2) + SAM head + no_mem_embed.

    When ``load_weights=True`` without explicit ``dtype``/``precision``, applies
    permanent :data:`~sam3.dtype_policy.DEFAULT_INFERENCE_DTYPE` (bf16)
    after load. RoPE tables stay float32.
    """
    precision = resolve_precision(
        precision,
        dtype=dtype,
        use_autocast=use_autocast,
        load_weights=load_weights,
    )
    device = resolve_device(device, precision=precision)
    # Nested backbone: structure only; cast happens on the full predictor.
    backbone = build_production_vision_backbone(
        add_sam2_neck=True,
        precompute_pe=precompute_pe,
        **vit_overrides,
    )
    head = build_production_sam_head()
    pred = SamInteractivePredictor(backbone=backbone, head=head, precision=precision)
    if load_weights:
        load_interactive_weights(pred, checkpoint_path=checkpoint_path)
    finalize_module(pred, precision, device=device)
    pred.eval()
    return pred


def load_interactive_weights(
    predictor: SamInteractivePredictor,
    ckpt: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    checkpoint_path: Optional[str] = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load backbone + SAM head + tracker ``no_mem_embed`` into predictor."""
    if ckpt is None:
        ckpt = load_sam3_checkpoint(checkpoint_path)

    miss_b, skip_b = load_vision_backbone_weights(
        predictor.backbone, ckpt, strict=False
    )
    miss_h, skip_h = load_sam_head_weights(predictor.head, ckpt, strict=False)

    key = "tracker.no_mem_embed"
    if key not in ckpt:
        raise KeyError(f"missing {key} in checkpoint")
    with torch.no_grad():
        predictor.no_mem_embed.copy_(ckpt[key].to(predictor.no_mem_embed.dtype))

    missing = [f"backbone.{k}" for k in miss_b] + [f"head.{k}" for k in miss_h]
    skipped = list(skip_b) + list(skip_h)
    if strict and missing:
        raise RuntimeError(f"strict load failed: {missing[:15]=}")
    return missing, skipped


def summarize_load(module: nn.Module, loaded_keys: Iterable[str]) -> dict[str, Any]:
    total = len(module.state_dict())
    loaded = len(list(loaded_keys))
    return {"module_params": total, "loaded_keys": loaded}


# ---------------------------------------------------------------------------
# Text / open-vocab detector (Sam3Image)
# ---------------------------------------------------------------------------


def _create_text_encoder(bpe_path: Optional[str] = None) -> VETextEncoder:
    path = bpe_path or DEFAULT_BPE_PATH
    tokenizer = SimpleTokenizer(bpe_path=path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_transformer_encoder() -> TransformerEncoderFusion:
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=nn.MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=nn.MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )
    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=False,  # eval path
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_geometry_encoder() -> SequenceGeometryEncoder:
    geo_pos_enc = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=nn.MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=nn.MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )
    return SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=False,
        add_cls=True,
        add_post_encode_proj=True,
    )


def _create_dot_product_scoring() -> DotProductScoring:
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head() -> UniversalSegmentationHead:
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
    )
    cross_attend_prompt = nn.MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )
    return UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=False,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def build_production_text_detector(
    *,
    bpe_path: Optional[str] = None,
    enable_segmentation: bool = True,
    add_sam2_neck: bool = False,
    precision: PrecisionConfig | None = None,
    dtype: torch.dtype | str | None = None,
    device: str | torch.device | None = None,
    load_weights: bool = False,
    checkpoint_path: Optional[str] = None,
    use_autocast: bool = False,
    **vit_overrides: Any,
) -> Sam3Image:
    """Build SAM3 image text detector (vision + text + DETR + seg).

    ``load_weights=True`` without dtype defaults to
    :data:`~sam3.dtype_policy.DEFAULT_INFERENCE_DTYPE` after load.
    """
    precision = resolve_precision(
        precision,
        dtype=dtype,
        use_autocast=use_autocast,
        load_weights=load_weights,
    )
    device = resolve_device(device, precision=precision)
    vision = build_production_vision_backbone(
        add_sam2_neck=add_sam2_neck,
        precompute_pe=False,
        **vit_overrides,
    )
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(visual=vision, text=text_encoder, scalp=1)
    transformer = TransformerWrapper(
        encoder=_create_transformer_encoder(),
        decoder=create_sam3_image_decoder(use_act_checkpoint=False),
        d_model=256,
    )
    model = Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=_create_segmentation_head() if enable_segmentation else None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=_create_dot_product_scoring(),
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
    )
    if load_weights:
        load_text_detector_weights(model, checkpoint_path=checkpoint_path)
    finalize_module(model, precision, device=device)
    model.eval()
    return model


def extract_detector_state_dict(
    ckpt: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map ``detector.*`` keys → ``Sam3Image`` module keys (strip prefix)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if not k.startswith(_DETECTOR_PREFIX):
            continue
        name = k[len(_DETECTOR_PREFIX) :]
        if ".presence_head.2." in name:
            name = name.replace(".presence_head.2.", ".presence_head.")
        # ViT freqs_cis complex → real/imag for our trunk
        if name.endswith("attn.freqs_cis") and torch.is_complex(v):
            base = name[: -len("freqs_cis")]
            out[base + "freqs_cis_real"] = v.real.contiguous()
            out[base + "freqs_cis_imag"] = v.imag.contiguous()
            continue
        out[name] = v
    if not out:
        raise KeyError("No detector.* keys found in checkpoint")
    return out


def load_text_detector_weights(
    model: Sam3Image,
    ckpt: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    checkpoint_path: Optional[str] = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load ``detector.*`` weights into ``Sam3Image``."""
    if ckpt is None:
        ckpt = load_sam3_checkpoint(checkpoint_path)
    det = extract_detector_state_dict(ckpt)
    filtered = _filter_state_dict_for_module(model, det)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    skipped = [k for k in det if k not in filtered]
    if strict and (missing or skipped):
        raise RuntimeError(
            f"strict load failed: missing={missing[:20]} skipped={skipped[:20]}"
        )
    return list(missing), skipped + list(unexpected)


def build_production_text_predictor(
    *,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    load_weights: bool = True,
    checkpoint_path: Optional[str] = None,
    precision: PrecisionConfig | None = None,
    dtype: torch.dtype | str | None = None,
    use_autocast: bool = False,
    **build_kwargs: Any,
) -> Sam3TextPredictor:
    """Build + optionally load weights for the text open-vocab API.

    Defaults (``load_weights=True``): permanent
    :data:`~sam3.dtype_policy.DEFAULT_INFERENCE_DTYPE` on CUDA.
    """
    for key in (
        "precision",
        "dtype",
        "device",
        "load_weights",
        "checkpoint_path",
        "use_autocast",
    ):
        build_kwargs.pop(key, None)

    precision = resolve_precision(
        precision,
        dtype=dtype,
        use_autocast=use_autocast,
        load_weights=load_weights,
    )
    # Preserve historical default device="cuda" even without precision.
    if device == "cuda" and not torch.cuda.is_available():
        target_device: str | torch.device | None = "cpu"
    else:
        target_device = resolve_device(device, precision=precision) or device

    model = build_production_text_detector(
        load_weights=load_weights,
        checkpoint_path=checkpoint_path,
        precision=precision,
        device=target_device,
        **build_kwargs,
    )
    model.eval()
    return Sam3TextPredictor(
        model=model,
        device=str(target_device),
        confidence_threshold=confidence_threshold,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# Video tracker (SAM2-style memory attention)
# ---------------------------------------------------------------------------


def build_production_tracker(
    *,
    with_backbone: bool = False,
    precision: PrecisionConfig | None = None,
    dtype: torch.dtype | str | None = None,
    device: str | torch.device | None = None,
    load_weights: bool = False,
    checkpoint_path: Optional[str] = None,
    use_autocast: bool = False,
    **tracker_kwargs: Any,
) -> Sam3Tracker:
    """Production tracker matching ``build_tracker`` (no temporal selection)."""
    precision = resolve_precision(
        precision,
        dtype=dtype,
        use_autocast=use_autocast,
        load_weights=load_weights,
    )
    device = resolve_device(device, precision=precision)
    backbone = None
    if with_backbone:
        # Official tracker uses SAM3VLBackbone(scalp=1, text=None) so that
        # ``forward_image`` returns ``sam2_backbone_out`` for high-res SAM heads.
        visual = build_production_vision_backbone(
            add_sam2_neck=True,
            precompute_pe=False,
        )
        backbone = SAM3VLBackbone(visual=visual, text=None, scalp=1)
    tracker = build_sam3_tracker(
        backbone=backbone,
        precision=precision,
        **tracker_kwargs,
    )
    if load_weights:
        load_tracker_weights(tracker, checkpoint_path=checkpoint_path)
    finalize_module(tracker, precision, device=device)
    tracker.eval()
    return tracker


def extract_tracker_state_dict(
    ckpt: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map ``tracker.*`` → ``Sam3Tracker`` keys (strip prefix)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if not k.startswith(_TRACKER_PREFIX):
            continue
        name = k[len(_TRACKER_PREFIX) :]
        if name.startswith("maskmem_backbone.fuser.layers."):
            name = (
                name.replace(".dwconv.", ".conv_dw.")
                .replace(".pwconv1.", ".mlp.fc1.")
                .replace(".pwconv2.", ".mlp.fc2.")
            )
        # Official complex RoPE freqs → real/imag if present on tracker RoPE
        if name.endswith("freqs_cis") and torch.is_complex(v):
            base = name[: -len("freqs_cis")]
            out[base + "freqs_cis_real"] = v.real.contiguous()
            out[base + "freqs_cis_imag"] = v.imag.contiguous()
            continue
        out[name] = v
    if not out:
        raise KeyError("No tracker.* keys found in checkpoint")
    return out


def load_tracker_weights(
    tracker: nn.Module,
    ckpt: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    checkpoint_path: Optional[str] = None,
    strict: bool = False,
    load_backbone: bool = True,
) -> tuple[list[str], list[str]]:
    """Load ``tracker.*`` (and optional detector vision trunk) into tracker.

    When ``tracker.backbone`` is a ``SAM3VLBackbone``, also load
    ``detector.backbone.vision_backbone.*`` into ``backbone.vision_backbone``.
    """
    if ckpt is None:
        ckpt = load_sam3_checkpoint(checkpoint_path)
    trk = extract_tracker_state_dict(ckpt)
    filtered = _filter_state_dict_for_module(tracker, trk)
    missing, unexpected = tracker.load_state_dict(filtered, strict=False)
    skipped = [k for k in trk if k not in filtered]

    if (
        load_backbone
        and getattr(tracker, "backbone", None) is not None
        and hasattr(tracker.backbone, "vision_backbone")
    ):
        miss_b, skip_b = load_vision_backbone_weights(
            tracker.backbone.vision_backbone, ckpt, strict=False
        )
        missing = list(missing) + [f"backbone.vision_backbone.{k}" for k in miss_b]
        skipped = list(skipped) + list(skip_b)

    if strict and (missing or skipped):
        raise RuntimeError(
            f"strict load failed: missing={missing[:20]} skipped={skipped[:20]}"
        )
    return list(missing), skipped + list(unexpected)


def build_production_video_tracker(
    *,
    load_weights: bool = True,
    checkpoint_path: Optional[str] = None,
    with_backbone: bool = True,
    precision: PrecisionConfig | None = None,
    dtype: torch.dtype | str | None = None,
    device: str | torch.device | None = None,
    use_autocast: bool = False,
    **kwargs: Any,
) -> Sam3VideoTracker:
    """Build multi-frame video tracker shell with optional real weights.

    Default ``with_backbone=True`` so JPEG / tensor frames can be encoded.
    Default ``load_weights=True`` → permanent DEFAULT_INFERENCE_DTYPE on CUDA.
    """
    precision = resolve_precision(
        precision,
        dtype=dtype,
        use_autocast=use_autocast,
        load_weights=load_weights,
    )
    device = resolve_device(device, precision=precision)
    # Nested builder loads + casts when precision/device given.
    kwargs.pop("precision", None)
    kwargs.pop("dtype", None)
    kwargs.pop("device", None)
    kwargs.pop("use_autocast", None)
    kwargs.pop("load_weights", None)
    kwargs.pop("checkpoint_path", None)
    tracker = build_production_tracker(
        with_backbone=with_backbone,
        load_weights=load_weights,
        checkpoint_path=checkpoint_path,
        precision=precision,
        device=device,
        **kwargs,
    )
    tracker.eval()
    return Sam3VideoTracker(tracker=tracker, precision=precision)


def build_production_text_on_video(
    *,
    bpe_path: Optional[str] = None,
    load_weights: bool = True,
    checkpoint_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    recondition_every_nth_frame: int = 16,
    precision: PrecisionConfig | None = None,
    dtype: torch.dtype | str | None = None,
    device: str | torch.device | None = None,
    use_autocast: bool = False,
) -> Sam3TextOnVideo:
    """Build shared-backbone text-on-video model (detector + tracker).

    Vision DualViTDet neck is shared so the ViT is not duplicated in memory.
    Default ``load_weights=True`` → permanent DEFAULT_INFERENCE_DTYPE on CUDA.
    """
    precision = resolve_precision(
        precision,
        dtype=dtype,
        use_autocast=use_autocast,
        load_weights=load_weights,
    )
    device = resolve_device(device, precision=precision)
    visual = build_production_vision_backbone(
        add_sam2_neck=True,
        precompute_pe=False,
    )
    text_encoder = _create_text_encoder(bpe_path)
    # Detector uses text; tracker reuses same visual trunk/neck
    det_backbone = SAM3VLBackbone(visual=visual, text=text_encoder, scalp=1)
    trk_backbone = SAM3VLBackbone(visual=visual, text=None, scalp=1)

    transformer = TransformerWrapper(
        encoder=_create_transformer_encoder(),
        decoder=create_sam3_image_decoder(use_act_checkpoint=False),
        d_model=256,
    )
    detector = Sam3Image(
        backbone=det_backbone,
        transformer=transformer,
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=_create_segmentation_head(),
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=_create_dot_product_scoring(),
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
    )
    tracker = build_sam3_tracker(
        backbone=trk_backbone,
        precision=precision,
    )

    if load_weights:
        # Load detector.* (includes shared vision under backbone.vision_backbone)
        load_text_detector_weights(detector, checkpoint_path=checkpoint_path)
        # Tracker heads / memory (vision already loaded via detector)
        load_tracker_weights(
            tracker, checkpoint_path=checkpoint_path, load_backbone=False
        )

    model = Sam3TextOnVideo(
        detector=detector,
        tracker=tracker,
        confidence_threshold=confidence_threshold,
        recondition_every_nth_frame=recondition_every_nth_frame,
        precision=precision,
    )
    finalize_module(model, precision, device=device)
    model.eval()
    return model
