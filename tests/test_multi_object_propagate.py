"""Targeted tests for one-frame multi-object propagation."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
import torch.nn as nn

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for multi-object propagate tests", allow_module_level=True
    )

from sam3.tracking.multi_object_propagate import propagate_objects_one_frame
from sam3.tracking.sam3_video_tracker import Sam3VideoTracker

DEVICE = torch.device("cuda")


class _FakeTracker(nn.Module):
    """Minimal tracker contract that records ``track_step`` calls."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))
        self.track_step_calls = 0

    def track_step(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: List[Tensor],
        current_vision_pos_embeds: List[Tensor],
        feat_sizes: List[tuple[int, int]],
        image: Tensor | None,
        point_inputs: Dict[str, Tensor] | None,
        mask_inputs: Tensor | None,
        output_dict: Dict[str, Dict[int, Dict[str, Any]]],
        num_frames: int,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Tensor | None = None,
        use_prev_mem_frame: bool = True,
    ) -> Dict[str, Any]:
        self.track_step_calls += 1
        return {
            "pred_masks": torch.randn(1, 1, 8, 8, device=DEVICE),
            "track_step_calls": self.track_step_calls,
            "run_mem_encoder": run_mem_encoder,
        }


def _make_states(
    *, num_objects: int, num_frames: int = 1
) -> tuple[Sam3VideoTracker, list[dict], _FakeTracker]:
    fake = _FakeTracker().to(DEVICE)
    tracker = Sam3VideoTracker(fake, image_size=64)
    images = torch.randn(
        num_frames, 3, tracker.image_size, tracker.image_size, device=DEVICE
    )
    states = []
    for _ in range(num_objects):
        state = tracker.init_state(
            images=images,
            video_height=tracker.image_size,
            video_width=tracker.image_size,
            device=DEVICE,
        )
        states.append(state)
    return tracker, states, fake


def _install_counted_encoder(
    tracker: Sam3VideoTracker,
) -> dict[str, int]:
    counters = {"encode_calls": 0}

    def _counted_encode(state_in: dict[str, Any], frame_idx: int):
        counters["encode_calls"] += 1
        feats: dict[str, Any] = {
            "vision_feats": [torch.randn(4, 1, 8, device=DEVICE)],
            "vision_pos_embeds": [torch.randn(4, 1, 8, device=DEVICE)],
            "feat_sizes": [(2, 2)],
            "image": state_in["images"][frame_idx : frame_idx + 1],
        }
        state_in["vision_features"][frame_idx] = feats
        return feats

    tracker._encode_frame_into_cache = _counted_encode  # type: ignore[assignment]
    return counters


def test_propagate_one_frame_encodes_once_for_multi_object():
    tracker, states, fake = _make_states(num_objects=3)
    counters = _install_counted_encoder(tracker)

    shared_cache: dict = {}
    outputs = propagate_objects_one_frame(
        tracker,
        states,
        frame_idx=0,
        shared_vision_cache=shared_cache,
        run_mem_encoder=True,
    )

    assert len(outputs) == 3
    assert all(out is not None for out in outputs)
    assert counters["encode_calls"] == 1
    # Sprint 8: B-batch path → one track_step with B=3 (not 3 serial calls)
    assert fake.track_step_calls == 1
    assert states[0]["vision_features"] is shared_cache
    assert states[1]["vision_features"] is shared_cache
    assert states[2]["vision_features"] is shared_cache


def test_propagate_objects_one_frame_skips_precomputed_outputs():
    tracker, states, fake = _make_states(num_objects=3)
    counters = _install_counted_encoder(tracker)

    states[0]["output_dict"]["cond_frame_outputs"][0] = {"tag": "cond"}
    states[1]["output_dict"]["non_cond_frame_outputs"][0] = {"tag": "non_cond"}

    outputs = propagate_objects_one_frame(
        tracker, states, frame_idx=0, shared_vision_cache={}
    )

    assert outputs[0] is states[0]["output_dict"]["cond_frame_outputs"][0]
    assert outputs[1] is states[1]["output_dict"]["non_cond_frame_outputs"][0]
    assert outputs[2] is not None
    assert fake.track_step_calls == 1
    assert counters["encode_calls"] == 1
