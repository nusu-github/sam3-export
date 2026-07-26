"""One-frame multi-object propagation with a shared vision-feature cache."""

from __future__ import annotations

from typing import Any

from tensordict import TensorDictBase
from torch import Tensor

from .td_features import unpack_frame_features


def _repeat_batch(value: Tensor, batch: int, *, dimension: int) -> Tensor:
    repeats = [1] * value.ndim
    repeats[dimension] = batch
    return value.repeat(*repeats)


def _slice_output(value: Any, index: int) -> Any:
    if isinstance(value, Tensor):
        if value.ndim > 0 and value.shape[0] > index:
            return value[index : index + 1]
        return value
    if isinstance(value, TensorDictBase):
        return value[index : index + 1]
    if isinstance(value, dict):
        return {key: _slice_output(item, index) for key, item in value.items()}
    if isinstance(value, list):
        return [_slice_output(item, index) for item in value]
    return value


def propagate_objects_one_frame(
    video_tracker: Any,
    object_states: list[dict],
    frame_idx: int,
    *,
    shared_vision_cache: dict | None = None,
    run_mem_encoder: bool = True,
) -> list[dict | None]:
    """Run one-frame propagation for multiple object states.

    Args:
        video_tracker: ``Sam3VideoTracker``-like object.
        object_states: List of SAM3 inference states, one per object.
        frame_idx: Frame index to process.
        shared_vision_cache: Optional dict shared by all states for cached feats.
        run_mem_encoder: Whether to call ``track_step(..., run_mem_encoder=...)``.

    Returns:
        List aligned with ``object_states`` containing each object's current output
        dict (or existing out when the frame was already tracked).
    """

    if shared_vision_cache is not None:
        attach_shared = getattr(video_tracker, "attach_shared_vision_cache", None)
        for state in object_states:
            if callable(attach_shared):
                attach_shared(state, shared_vision_cache)
            else:
                state["vision_features"] = shared_vision_cache

    outputs: list[dict | None] = [None] * len(object_states)
    pending_idxs: list[int] = []

    for i, state in enumerate(object_states):
        out_dict = state["output_dict"]
        current_out = out_dict["cond_frame_outputs"].get(frame_idx)
        if current_out is None:
            current_out = out_dict["non_cond_frame_outputs"].get(frame_idx)
        if current_out is None:
            pending_idxs.append(i)
        else:
            outputs[i] = current_out

    if not pending_idxs:
        return outputs

    store_output = getattr(video_tracker, "_store_frame_output", None)

    # Encode once, then amortize the independent per-object work in one B launch.
    first_state = object_states[pending_idxs[0]]
    frame = video_tracker._get_frame_features(first_state, frame_idx)
    vision_feats, vision_pos, feat_sizes, image = unpack_frame_features(
        frame, device=first_state["device"]
    )
    batch = len(pending_idxs)
    batched_feats = [_repeat_batch(value, batch, dimension=1) for value in vision_feats]
    batched_pos = [_repeat_batch(value, batch, dimension=1) for value in vision_pos]
    batched_image = None
    if image is not None:
        batched_image = _repeat_batch(image, batch, dimension=0)

    # The legacy helper predates BaseVideoStateV1. Its batched path is used only
    # when every pending object has the same empty/compatible state; the M4
    # runtime performs the complete per-object state packing contract.
    output_dict = first_state["output_dict"]
    out = video_tracker.tracker.track_step(
        frame_idx=frame_idx,
        is_init_cond_frame=False,
        current_vision_feats=batched_feats,
        current_vision_pos_embeds=batched_pos,
        feat_sizes=feat_sizes,
        image=batched_image,
        point_inputs=None,
        mask_inputs=None,
        output_dict=output_dict,
        num_frames=first_state["num_frames"],
        track_in_reverse=False,
        run_mem_encoder=run_mem_encoder,
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
    )

    for output_index, i in enumerate(pending_idxs):
        state = object_states[i]
        object_out = _slice_output(out, output_index)
        if callable(store_output):
            store_output(state, "non_cond_frame_outputs", frame_idx, object_out)
        else:
            state["output_dict"]["non_cond_frame_outputs"][frame_idx] = object_out
        state["frames_already_tracked"][frame_idx] = {"reverse": False}
        outputs[i] = object_out

    return outputs
