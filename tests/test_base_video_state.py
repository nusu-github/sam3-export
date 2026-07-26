"""M4 BaseVideoStateV1 host-state contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from sam3.runtime.base_video_state import (
    BaseVideoStateV1,
    BaseVideoVariantParameters,
    StateCapacityError,
    VideoStateEntry,
)


def _entry(frame_index: int, *, conditioning: bool) -> VideoStateEntry:
    value = np.asarray([frame_index], dtype=np.float32)
    return VideoStateEntry(
        frame_index=frame_index,
        conditioning=conditioning,
        memory_features=("memory", frame_index),
        memory_position=("position", frame_index),
        object_pointer=("pointer", frame_index),
        mask=np.zeros((4, 4), dtype=np.bool_),
        score=float(frame_index),
        low_res_logits=value,
        object_score=1.0,
    )


def _variant_manifest() -> dict:
    values = {
        "num-maskmem": 7,
        "conditioning-spatial-capacity": 4,
        "non-conditioning-spatial-capacity": 6,
        "total-spatial-input-capacity": 10,
        "object-pointer-capacity": 16,
        "hidden-dimension": 256,
        "memory-dimension": 64,
        "memory-spatial-size": [72, 72],
        "temporal-stride": 1,
        "memory-sigmoid-scale": 20.0,
        "memory-sigmoid-bias": -10.0,
        "non-overlap-memory": False,
    }
    return {
        "model": {
            "variant_parameters": [
                {"name": name, "value": value} for name, value in values.items()
            ]
        }
    }


def test_variant_parameters_are_required_and_exact() -> None:
    values = BaseVideoVariantParameters.from_manifest(_variant_manifest())
    assert values.memory_spatial_size == (72, 72)
    manifest = _variant_manifest()
    manifest["model"]["variant_parameters"][0]["value"] = 8
    with pytest.raises(ValueError, match="num_maskmem"):
        BaseVideoVariantParameters.from_manifest(manifest)


def test_conditioning_capacity_replaces_same_frame_without_eviction() -> None:
    state = BaseVideoStateV1(batch_capacity=4)
    state.add_object(7)
    for frame_index in range(4):
        state.commit(7, _entry(frame_index, conditioning=True))
    state.commit(7, _entry(3, conditioning=True))
    assert list(state.require_object(7).conditioning) == [0, 1, 2, 3]
    with pytest.raises(StateCapacityError, match="conditioning capacity"):
        state.commit(7, _entry(4, conditioning=True))


def test_pack_memory_zero_one_max_and_padding() -> None:
    state = BaseVideoStateV1(batch_capacity=4)
    state.add_object(10)
    state.add_object(20)
    (empty,) = state.pack([10], frame_index=5, reverse=False)
    assert empty.object_valid.tolist() == [True, False, False, False]
    assert not empty.memory_valid.any()
    assert not empty.pointer_valid.any()

    state.commit(10, _entry(0, conditioning=True))
    (one,) = state.pack([10], frame_index=1, reverse=False)
    assert one.memory_valid[0].sum() == 1
    assert one.memory_age[0, 0] == 1
    assert one.memory_conditioning[0, 0]
    assert one.pointer_valid[0].sum() == 1
    assert one.pointer_tpos_denominator.tolist() == [1.0] * 4

    (same_frame,) = state.pack([10], frame_index=0, reverse=False)
    assert not same_frame.memory_valid.any()
    assert not same_frame.pointer_valid.any()

    for frame_index in (10, 20, 30):
        state.commit(10, _entry(frame_index, conditioning=True))
    for frame_index in range(25, 40):
        state.commit(10, _entry(frame_index, conditioning=False))
    (full,) = state.pack([10, 20], frame_index=40, reverse=False)
    assert full.memory_valid[0].sum() == 10
    assert full.memory_conditioning[0, :4].all()
    assert not full.memory_conditioning[0, 4:].any()
    assert full.memory_age[0, 4:].tolist() == [1, 2, 3, 4, 5, 6]
    assert full.pointer_valid[0].sum() == 16
    assert not full.memory_valid[1].any()


def test_pack_chunks_capacity_plus_one_and_reverse_signed_age() -> None:
    state = BaseVideoStateV1(batch_capacity=4)
    for object_id in range(5):
        state.add_object(object_id)
    state.commit(0, _entry(8, conditioning=True))
    chunks = state.pack(
        list(range(5)), frame_index=5, reverse=True, video_frame_count=20
    )
    assert len(chunks) == 2
    assert chunks[0].object_valid.sum() == 4
    assert chunks[1].object_valid.sum() == 1
    assert chunks[0].memory_age[0, 0] == -3
    assert chunks[0].pointer_tpos_denominator.tolist() == [15.0] * 4


def test_correction_replacement_invalidates_only_influence_direction() -> None:
    state = BaseVideoStateV1(batch_capacity=4)
    state.add_object(1)
    state.commit(1, _entry(2, conditioning=True))
    for frame_index in (3, 4, 5):
        state.commit(1, _entry(frame_index, conditioning=False))
    before = state.revision
    state.commit(1, _entry(4, conditioning=True))
    object_state = state.require_object(1)
    assert list(object_state.non_conditioning) == [3]
    assert object_state.conditioning[4].conditioning
    assert state.revision == before + 1


def test_duplicate_and_unknown_objects_are_rejected() -> None:
    state = BaseVideoStateV1(batch_capacity=8)
    state.add_object(3)
    with pytest.raises(ValueError, match="duplicate"):
        state.add_object(3)
    with pytest.raises(KeyError, match="unknown"):
        state.pack([4], frame_index=0, reverse=False)


def test_b8_candidate_pads_and_chunks_capacity_plus_one() -> None:
    state = BaseVideoStateV1(batch_capacity=8)
    for object_id in range(9):
        state.add_object(object_id)
    chunks = state.pack(list(range(9)), frame_index=0, reverse=False)
    assert [int(chunk.object_valid.sum()) for chunk in chunks] == [8, 1]
    assert all(chunk.object_valid.shape == (8,) for chunk in chunks)
