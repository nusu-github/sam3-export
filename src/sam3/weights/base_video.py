"""Checkpoint adapter for the M4 SAM3 base video production components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sam3.export.base_video import (
    BaseMemoryCommit,
    BaseTrackerPreviewMultimask3,
    BaseTrackerPreviewSingle1,
    BaseTrackerStepAndCommitSingle1,
    TrackerFrameEncode,
)
from sam3.runtime.base_video_state import BaseVideoVariantParameters

from .load_sam3 import (
    build_production_tracker,
    load_sam3_checkpoint,
)


@dataclass(frozen=True)
class BaseVideoModules:
    tracker: nn.Module
    variant: BaseVideoVariantParameters
    frame_encode: TrackerFrameEncode
    preview_multimask3: BaseTrackerPreviewMultimask3
    preview_single1: BaseTrackerPreviewSingle1
    memory_commit: BaseMemoryCommit
    step_and_commit_single1: BaseTrackerStepAndCommitSingle1


def base_video_variant_from_checkpoint(
    checkpoint: dict[str, torch.Tensor], tracker: nn.Module
) -> BaseVideoVariantParameters:
    """Read serialized dimensions and verify official-builder scalar policy."""

    required_shapes = {
        "tracker.maskmem_tpos_enc": (7, 1, 1, 64),
        "tracker.no_mem_embed": (1, 1, 256),
        "tracker.no_mem_pos_enc": (1, 1, 256),
        "tracker.no_obj_ptr": (1, 256),
        "tracker.no_obj_embed_spatial": (1, 64),
    }
    for name, expected in required_shapes.items():
        value = checkpoint.get(name)
        if value is None:
            raise KeyError(f"checkpoint is missing M4 parameter: {name}")
        if tuple(value.shape) != expected:
            raise ValueError(
                f"checkpoint M4 parameter shape mismatch for {name}: "
                f"{tuple(value.shape)} != {expected}"
            )

    variant = BaseVideoVariantParameters(
        num_maskmem=int(tracker.num_maskmem),
        conditioning_spatial_capacity=int(tracker.max_cond_frames_in_attn),
        non_conditioning_spatial_capacity=int(tracker.num_maskmem - 1),
        total_spatial_input_capacity=int(
            tracker.max_cond_frames_in_attn + tracker.num_maskmem - 1
        ),
        object_pointer_capacity=int(tracker.max_obj_ptrs_in_encoder),
        hidden_dimension=int(tracker.hidden_dim),
        memory_dimension=int(tracker.mem_dim),
        memory_spatial_size=(
            int(tracker.sam_image_embedding_size),
            int(tracker.sam_image_embedding_size),
        ),
        temporal_stride=int(tracker.memory_temporal_stride_for_eval),
        memory_sigmoid_scale=float(tracker.sigmoid_scale_for_mem_enc),
        memory_sigmoid_bias=float(tracker.sigmoid_bias_for_mem_enc),
        non_overlap_memory=bool(tracker.non_overlap_masks_for_mem_enc),
    )
    variant.validate()
    return variant


def build_base_video_modules(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "fp16",
) -> BaseVideoModules:
    """Build only the fixed SAM3-base modules used by the M4 release plan."""

    checkpoint = load_sam3_checkpoint(str(checkpoint_path))
    tracker = build_production_tracker(
        with_backbone=True,
        load_weights=True,
        checkpoint_path=str(checkpoint_path),
        device=device,
        dtype=dtype,
        num_maskmem=7,
        max_cond_frames_in_attn=4,
        max_obj_ptrs_in_encoder=16,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    ).eval()
    variant = base_video_variant_from_checkpoint(checkpoint, tracker)
    frame = TrackerFrameEncode(tracker, use_cuda_autocast=True).eval()
    multimask = BaseTrackerPreviewMultimask3(
        tracker, variant, use_cuda_autocast=True
    ).eval()
    single = BaseTrackerPreviewSingle1(tracker, variant, use_cuda_autocast=True).eval()
    commit = BaseMemoryCommit(tracker, variant, use_cuda_autocast=True).eval()
    fused = BaseTrackerStepAndCommitSingle1(single, commit).eval()
    return BaseVideoModules(
        tracker=tracker,
        variant=variant,
        frame_encode=frame,
        preview_multimask3=multimask,
        preview_single1=single,
        memory_commit=commit,
        step_and_commit_single1=fused,
    )


__all__ = [
    "BaseVideoModules",
    "base_video_variant_from_checkpoint",
    "build_base_video_modules",
]
