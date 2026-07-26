"""Exact checkpoint mapping for the SAM3.1 native Multiplex components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn

from sam3.export.multiplex_video import (
    MultiplexFrameEncode,
    MultiplexInteractionPreviewMultimask3,
    MultiplexInteractionPreviewSingle1,
    MultiplexMemoryCommit,
    MultiplexPropagation,
    MultiplexScatterReplaceCommit,
)
from sam3.primitives.mlp import MLP
from sam3.primitives.position_encoding import PositionEmbeddingSine
from sam3.runtime.multiplex_state import MultiplexVariantParameters
from sam3.tracking.memory import create_maskmem_backbone
from sam3.tracking.multiplex_transformer import create_multiplex_transformer
from sam3.vision.multiplex_mask_decoder import create_multiplex_mask_decoder
from sam3.vision.necks import Sam3TriViTDetNeck
from sam3.vision.prompt_encoder import PositionEmbeddingRandom

from .load_sam3 import build_production_sam_head, build_production_vit

SAM31_REPOSITORY: Final[str] = "facebook/sam3.1"
SAM31_REVISION: Final[str] = "daa63191845a41281374e725f4c9e51c7a824460"
SAM31_CHECKPOINT_NAME: Final[str] = "sam3.1_multiplex.pt"
SAM31_CHECKPOINT_SHA256: Final[str] = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)
TRI_NECK_PREFIX: Final[str] = "detector.backbone.vision_backbone."
TRACKER_PREFIX: Final[str] = "tracker.model."


@dataclass(frozen=True)
class ParameterMappingReport:
    """One-to-one target mapping, including value identity."""

    checkpoint_prefix: str
    checkpoint_key_count: int
    module_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    value_mismatches: tuple[str, ...]

    @property
    def exact(self) -> bool:
        return not (
            self.missing_keys
            or self.unexpected_keys
            or self.shape_mismatches
            or self.value_mismatches
        )

    def require_exact(self) -> None:
        if not self.exact:
            raise RuntimeError(
                "SAM3.1 checkpoint mapping is not exact: "
                f"missing={list(self.missing_keys)}, "
                f"unexpected={list(self.unexpected_keys)}, "
                f"shape={list(self.shape_mismatches)}, "
                f"value={list(self.value_mismatches)}"
            )


class MultiplexTrackerCore(nn.Module):
    """Local learned SAM3.1 tracker modules with checkpoint-native names."""

    def __init__(self) -> None:
        super().__init__()
        interactive = build_production_sam_head(iou_prediction_use_sigmoid=False)
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(7, 1, 1, 256))
        self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, 256))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(16, 256))
        self.output_valid_embed = nn.Parameter(torch.zeros(16, 256))
        self.output_invalid_embed = nn.Parameter(torch.zeros(16, 256))
        self.image_pe_layer = PositionEmbeddingRandom(128)
        self.interactive_mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.transformer = create_multiplex_transformer()
        self.maskmem_backbone = create_maskmem_backbone(
            out_dim=256,
            in_dim=256,
            fuser_layers=2,
            pe_dim=256,
            interpol_size=(1152, 1152),
            precompute_resolution=None,
            multiplex_count=16,
            starting_out_chan=4,
            input_channel_multiplier=2,
            official_fuser_names=True,
        )
        self.interactive_sam_prompt_encoder = interactive.prompt_encoder
        self.interactive_sam_mask_decoder = interactive.mask_decoder
        self.sam_mask_decoder = create_multiplex_mask_decoder()
        self.obj_ptr_proj = MLP(256, 256, 256, 3)
        self.interactive_obj_ptr_proj = MLP(256, 256, 256, 3)
        self.obj_ptr_tpos_proj = nn.Linear(256, 256)
        self.no_obj_ptr_linear = nn.Linear(256, 256)


@dataclass(frozen=True)
class MultiplexVideoModules:
    """Canonical M5 modules; deployment recipes may compose them differently."""

    frame_encode: MultiplexFrameEncode
    preview_multimask3: MultiplexInteractionPreviewMultimask3
    preview_single1: MultiplexInteractionPreviewSingle1
    propagation_bucket1: MultiplexPropagation
    propagation_bucket2: MultiplexPropagation
    propagation_dynamic: MultiplexPropagation
    memory_commit_bucket1: MultiplexMemoryCommit
    memory_commit_bucket2: MultiplexMemoryCommit
    memory_commit_dynamic: MultiplexMemoryCommit
    scatter_commit_bucket1: MultiplexScatterReplaceCommit
    scatter_commit_bucket2: MultiplexScatterReplaceCommit
    scatter_commit_dynamic: MultiplexScatterReplaceCommit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sam31_multiplex_checkpoint(path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    environment = os.environ.get("SAM31_MULTIPLEX_CHECKPOINT")
    if environment:
        candidates.append(Path(environment))
    for root in (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
        Path("/workspace/.cache/huggingface"),
    ):
        candidates.append(
            root
            / "hub"
            / "models--facebook--sam3.1"
            / "snapshots"
            / SAM31_REVISION
            / SAM31_CHECKPOINT_NAME
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "SAM3.1 Multiplex checkpoint is unavailable at the fixed revision; "
        "pass a path or set SAM31_MULTIPLEX_CHECKPOINT"
    )


def load_sam31_multiplex_checkpoint(
    path: str | Path | None = None,
    *,
    revision: str = SAM31_REVISION,
) -> dict[str, Tensor]:
    if revision != SAM31_REVISION:
        raise ValueError(f"SAM3.1 revision mismatch: {revision} != {SAM31_REVISION}")
    checkpoint_path = resolve_sam31_multiplex_checkpoint(path)
    actual_digest = _sha256(checkpoint_path)
    if actual_digest != SAM31_CHECKPOINT_SHA256:
        raise ValueError(
            "SAM3.1 checkpoint digest mismatch: "
            f"{actual_digest} != {SAM31_CHECKPOINT_SHA256}"
        )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if not isinstance(raw, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in raw.items()
    ):
        raise TypeError("SAM3.1 checkpoint must be a flat tensor state dictionary")
    return raw


def _mapped_state(checkpoint: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in checkpoint.items():
        if not name.startswith(prefix):
            continue
        target = name[len(prefix) :]
        if target.endswith("attn.freqs_cis") and torch.is_complex(value):
            base = target[: -len("freqs_cis")]
            result[base + "freqs_cis_real"] = value.real.contiguous()
            result[base + "freqs_cis_imag"] = value.imag.contiguous()
        else:
            result[target] = value
    if not result:
        raise KeyError(f"checkpoint has no target parameters under {prefix}")
    return result


def map_checkpoint_to_module(
    checkpoint: Mapping[str, Tensor],
    module: nn.Module,
    *,
    prefix: str,
    load: bool = True,
) -> ParameterMappingReport:
    """Compare and optionally load every target state value by name and shape."""

    expected = _mapped_state(checkpoint, prefix)
    actual_before = module.state_dict()
    missing = tuple(sorted(set(actual_before) - set(expected)))
    unexpected = tuple(sorted(set(expected) - set(actual_before)))
    common = sorted(set(actual_before) & set(expected))
    shape = tuple(
        name
        for name in common
        if tuple(actual_before[name].shape) != tuple(expected[name].shape)
    )
    if load and not (missing or unexpected or shape):
        module.load_state_dict(expected, strict=True)
    actual_after = module.state_dict()
    value = tuple(
        name
        for name in common
        if name not in shape
        and not torch.equal(actual_after[name].detach().cpu(), expected[name].cpu())
    )
    return ParameterMappingReport(
        checkpoint_prefix=prefix,
        checkpoint_key_count=len(expected),
        module_key_count=len(actual_before),
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=shape,
        value_mismatches=value,
    )


def build_sam31_tri_neck(
    checkpoint: Mapping[str, Tensor] | None = None,
) -> Sam3TriViTDetNeck:
    """Build the local Tri neck and require exact target-state coverage."""

    module = Sam3TriViTDetNeck(
        trunk=build_production_vit(),
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=None,
        ),
        d_model=256,
        scale_factors=(4.0, 2.0, 1.0),
    )
    if checkpoint is not None:
        report = map_checkpoint_to_module(
            checkpoint, module, prefix=TRI_NECK_PREFIX, load=True
        )
        report.require_exact()
    return module


def build_sam31_multiplex_tracker_core(
    checkpoint: Mapping[str, Tensor] | None = None,
) -> MultiplexTrackerCore:
    """Build all learned tracker targets and require exact checkpoint coverage."""

    module = MultiplexTrackerCore()
    if checkpoint is not None:
        report = map_checkpoint_to_module(
            checkpoint, module, prefix=TRACKER_PREFIX, load=True
        )
        report.require_exact()
    return module


def build_sam31_multiplex_video_modules(
    checkpoint_path: str | Path | None = None,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    use_cuda_autocast: bool = True,
) -> MultiplexVideoModules:
    """Build the M5 canonical set from the fixed checkpoint identity."""

    checkpoint = load_sam31_multiplex_checkpoint(checkpoint_path)
    tri_neck = build_sam31_tri_neck(checkpoint).eval().to(device=device, dtype=dtype)
    tracker = (
        build_sam31_multiplex_tracker_core(checkpoint)
        .eval()
        .to(device=device, dtype=dtype)
    )
    variant = MultiplexVariantParameters.native()
    return MultiplexVideoModules(
        frame_encode=MultiplexFrameEncode(
            tri_neck, tracker, use_cuda_autocast=use_cuda_autocast
        ).eval(),
        preview_multimask3=MultiplexInteractionPreviewMultimask3(
            tracker, variant, use_cuda_autocast=use_cuda_autocast
        ).eval(),
        preview_single1=MultiplexInteractionPreviewSingle1(
            tracker, variant, use_cuda_autocast=use_cuda_autocast
        ).eval(),
        propagation_bucket1=MultiplexPropagation(
            tracker,
            variant,
            bucket_count=1,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        propagation_bucket2=MultiplexPropagation(
            tracker,
            variant,
            bucket_count=2,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        propagation_dynamic=MultiplexPropagation(
            tracker,
            variant,
            bucket_count=None,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        memory_commit_bucket1=MultiplexMemoryCommit(
            tracker,
            variant,
            bucket_count=1,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        memory_commit_bucket2=MultiplexMemoryCommit(
            tracker,
            variant,
            bucket_count=2,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        memory_commit_dynamic=MultiplexMemoryCommit(
            tracker,
            variant,
            bucket_count=None,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        scatter_commit_bucket1=MultiplexScatterReplaceCommit(
            tracker,
            variant,
            bucket_count=1,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        scatter_commit_bucket2=MultiplexScatterReplaceCommit(
            tracker,
            variant,
            bucket_count=2,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
        scatter_commit_dynamic=MultiplexScatterReplaceCommit(
            tracker,
            variant,
            bucket_count=None,
            use_cuda_autocast=use_cuda_autocast,
        ).eval(),
    )


def verify_multiplex_checkpoint_shapes(checkpoint: Mapping[str, Tensor]) -> None:
    """Verify checkpoint-owned capacity/layout values without inventing defaults."""

    required = {
        "tracker.model.maskmem_tpos_enc": (7, 1, 1, 256),
        "tracker.model.no_obj_embed_spatial": (16, 256),
        "tracker.model.output_valid_embed": (16, 256),
        "tracker.model.output_invalid_embed": (16, 256),
        "tracker.model.sam_mask_decoder.iou_token.weight": (16, 256),
        "tracker.model.sam_mask_decoder.obj_score_token.weight": (16, 256),
        "tracker.model.sam_mask_decoder.mask_tokens.weight": (48, 256),
        "tracker.model.obj_ptr_tpos_proj.weight": (256, 256),
        "tracker.model.no_obj_ptr_linear.weight": (256, 256),
        "tracker.model.maskmem_backbone.mask_downsampler.encoder.0.weight": (
            16,
            32,
            3,
            3,
        ),
    }
    for name, expected_shape in required.items():
        value = checkpoint.get(name)
        if value is None:
            raise KeyError(f"checkpoint is missing SAM3.1 parameter: {name}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"SAM3.1 parameter shape mismatch for {name}: "
                f"{tuple(value.shape)} != {expected_shape}"
            )


__all__ = [
    "ParameterMappingReport",
    "SAM31_CHECKPOINT_NAME",
    "SAM31_CHECKPOINT_SHA256",
    "SAM31_REPOSITORY",
    "SAM31_REVISION",
    "TRACKER_PREFIX",
    "TRI_NECK_PREFIX",
    "MultiplexTrackerCore",
    "MultiplexVideoModules",
    "build_sam31_multiplex_tracker_core",
    "build_sam31_multiplex_video_modules",
    "build_sam31_tri_neck",
    "load_sam31_multiplex_checkpoint",
    "map_checkpoint_to_module",
    "resolve_sam31_multiplex_checkpoint",
    "verify_multiplex_checkpoint_shapes",
]
