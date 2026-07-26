"""One-frame multi-object propagation with a shared vision-feature cache."""

from __future__ import annotations

from typing import Any


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

    # Ensure vision features once before the serial per-object step.
    first_state = object_states[pending_idxs[0]]
    if hasattr(video_tracker, "ensure_frame_features"):
        video_tracker.ensure_frame_features(first_state, frame_idx)
    else:
        video_tracker._get_frame_features(first_state, frame_idx)

    for i in pending_idxs:
        state = object_states[i]
        out = video_tracker._run_track_step(
            state=state,
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            run_mem_encoder=run_mem_encoder,
            track_in_reverse=False,
        )
        if callable(store_output):
            store_output(state, "non_cond_frame_outputs", frame_idx, out)
        else:
            state["output_dict"]["non_cond_frame_outputs"][frame_idx] = out
        state["frames_already_tracked"][frame_idx] = {"reverse": False}
        outputs[i] = out

    return outputs
