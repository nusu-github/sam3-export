"""M5 Multiplex Public API lifecycle and private-state contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import typing

import numpy as np
from PIL import Image
import pytest

from sam3.runtime import (
    BASE_VIDEO_PLAN_ID,
    MULTIPLEX_VIDEO_PLAN_ID,
    InteractivePredictOptions,
    InteractivePrompt,
    ManifestError,
    MultiplexVideoSession,
    ObjectStateError,
    PlanNotFoundError,
    PreviewHandleError,
    SessionClosedError,
    VideoFramePrediction,
    VideoPrediction,
    VideoPreview,
    VideoStateError,
    create_multiplex_video_session,
    create_video_session,
)
from sam3.runtime.manifest import sha256_file
from sam3.runtime.multiplex_state import SlotAssignment
from sam3.runtime.multiplex_video import (
    _MultiplexBackendFrame,
    _MultiplexBackendPreview,
)

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "tests" / "fixtures" / "manifest_v2" / "minimal_valid.json"

VARIANT = {
    "bucket-capacity": 16,
    "max-buckets": 2,
    "num-maskmem": 7,
    "conditioning-spatial-capacity": 4,
    "non-conditioning-spatial-capacity": 6,
    "total-spatial-input-capacity": 10,
    "object-pointer-frame-capacity": 16,
    "hidden-dimension": 256,
    "memory-dimension": 256,
    "memory-spatial-size": [72, 72],
    "image-size": 1008,
    "mask-candidates": 3,
    "memory-mask-channels": 32,
    "memory-sigmoid-scale": 2.0,
    "memory-sigmoid-bias": -1.0,
    "condition-mask-foreground": 1.0,
    "condition-mask-background": 0.0,
    "non-overlap-memory": False,
}


class FakeMultiplexAdapter:
    def __init__(self, _plan: object) -> None:
        self.counters = {
            "frame_encodes": 0,
            "preview_launches": 0,
            "propagation_launches": 0,
            "memory_commits": 0,
            "assignment_updates": 0,
            "final_demuxes": 0,
            "state_demuxes": 0,
            "state_remuxes": 0,
            "d2h_bytes": 0,
            "state_d2h_bytes": 0,
        }
        self.slot_state = np.zeros((2, 16), dtype=np.float32)
        self.validity = np.zeros((2, 16), dtype=np.bool_)
        self.closed = False

    def reset_state(self) -> None:
        self.slot_state.fill(0)
        self.validity.fill(False)

    def add_slot(self, assignment: SlotAssignment) -> None:
        self.validity[assignment.bucket, assignment.slot] = True
        self.slot_state[assignment.bucket, assignment.slot] = 0
        self.counters["assignment_updates"] += 1

    def remove_slot(self, assignment: SlotAssignment) -> None:
        self.validity[assignment.bucket, assignment.slot] = False
        self.slot_state[assignment.bucket, assignment.slot] = 0
        self.counters["assignment_updates"] += 1

    def encode_frame(self, values: np.ndarray) -> object:
        self.counters["frame_encodes"] += 1
        return ("tri-frame", values.shape, self.counters["frame_encodes"])

    def preview(
        self,
        frame_cache: object,
        assignment: SlotAssignment,
        prompt_inputs: dict[str, np.ndarray],
        *,
        multimask: bool,
    ) -> _MultiplexBackendPreview:
        del frame_cache, prompt_inputs
        self.counters["preview_launches"] += 1
        count = 3 if multimask else 1
        value = float(assignment.bucket * 16 + assignment.slot + 1)
        logits = np.full((count, 288, 288), value, dtype=np.float32)
        logits[:, :, :144] *= -1
        scores = np.linspace(0.2, 0.8, count, dtype=np.float32)
        self.counters["d2h_bytes"] += logits.nbytes + scores.nbytes
        return _MultiplexBackendPreview(
            low_res_logits=logits,
            scores=scores,
            commit_mask=("mask", value),
            object_pointer=("pointer", value),
            object_score=("score", value),
        )

    def commit(
        self,
        frame_cache: object,
        assignment: SlotAssignment,
        preview: _MultiplexBackendPreview,
        *,
        frame_index: int,
    ) -> None:
        del frame_cache
        before = self.slot_state.copy()
        value = float(preview.low_res_logits[0, 0, -1] + frame_index)
        self.slot_state[assignment.bucket, assignment.slot] = value
        changed = before != self.slot_state
        expected = np.zeros_like(changed)
        expected[assignment.bucket, assignment.slot] = True
        assert np.array_equal(changed, expected)
        self.counters["memory_commits"] += 1

    def propagate(
        self,
        frame_cache: object,
        assignments: np.ndarray,
        *,
        frame_index: int,
        reverse: bool,
    ) -> _MultiplexBackendFrame:
        del frame_cache, reverse
        self.counters["propagation_launches"] += 1
        bucket_count = 2 if np.any(self.validity[1]) else 1
        self.slot_state[:bucket_count] += self.validity[:bucket_count] * (
            frame_index + 1
        )
        values = np.asarray(
            [self.slot_state[bucket, slot] for bucket, slot in assignments],
            dtype=np.float32,
        )
        logits = np.broadcast_to(
            values[:, None, None], (len(values), 288, 288)
        ).copy()
        scores = values.copy()
        self.counters["final_demuxes"] += 1
        self.counters["d2h_bytes"] += logits.nbytes + scores.nbytes
        return _MultiplexBackendFrame(low_res_logits=logits, scores=scores)

    def close(self) -> None:
        self.closed = True


def _write_bundle(path: Path, *, variant: dict[str, object] = VARIANT) -> Path:
    manifest = json.loads(SKELETON.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        target = path / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(record["id"].encode())
        record["size_bytes"] = target.stat().st_size
        record["digest"]["value"] = sha256_file(target)
    manifest["manifest_id"] = f"{MULTIPLEX_VIDEO_PLAN_ID}-manifest-v2"
    manifest["scope"].update(
        {
            "scope_label": (
                "SAM3.1 multiplex video tracking / point-box-mask correction / "
                "bucket16 / ORT CUDA v1"
            ),
            "use_case": "multiplex-video-tracking",
            "prompt_coverage": ["point", "box", "mask"],
            "capabilities": ["multiplex video tracking", "bucket16"],
            "exclusions": [
                "SAM3 base",
                "text/geometry PCS",
                "streaming",
                "CPU fallback",
            ],
        }
    )
    manifest["plan"].update(
        {
            "id": MULTIPLEX_VIDEO_PLAN_ID,
            "semantic_graph_kind": "sam3.1-native-multiplex-video-tracking",
            "components": [
                "MultiplexFrameEncode",
                "MultiplexPropagation",
                "MultiplexScatterReplaceCommit",
            ],
        }
    )
    manifest["model"].update(
        {
            "family": "sam3.1",
            "variant": "sam3.1_multiplex",
            "vision_layout": "tri-tracking-1008",
            "tracking_layout": "MultiplexStateV1",
            "model_revision": "daa63191845a41281374e725f4c9e51c7a824460",
            "variant_parameters": [
                {"name": name, "value": value} for name, value in variant.items()
            ],
        }
    )
    manifest["profile"]["id"] = "bucket16-max2-1008-p16-box1-mask288-fp16"
    manifest["profile"]["static_values"].extend(
        [
            {"name": "bucket-capacity", "value": 16},
            {"name": "maximum-buckets", "value": 2},
        ]
    )
    manifest["caches"][0].update(
        {
            "id": "multiplex-frame-cache",
            "lifetime": "session",
            "key_parts": [
                "frame-digest",
                "video-frame-identity",
                "original-size",
                "checkpoint-digest",
                "profile-id",
                "condition-id",
            ],
            "invalidated_by": ["video-change", "checkpoint-change"],
            "state_compatibility": "sam3-1-tri-interactive-propagation-v1",
        }
    )
    manifest_dir = path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / f"{MULTIPLEX_VIDEO_PLAN_ID}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    result = _write_bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        "sam3.runtime.multiplex_video._multiplex_video_adapter_factory",
        FakeMultiplexAdapter,
    )
    return result


def _frames(count: int = 4) -> list[Image.Image]:
    return [Image.new("RGB", (20, 10), (index, 0, 0)) for index in range(count)]


def _single_options() -> InteractivePredictOptions:
    return InteractivePredictOptions(multimask_output=False)


def test_preview_commit_is_single_slot_and_assignment_revision_is_stable(
    bundle: Path,
) -> None:
    session = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle
    )
    session.set_video(_frames())
    session.add_object(7)
    revision = session._state.revision
    first = session.preview(7, 0, options=_single_options())
    second = session.preview(7, 0, options=_single_options())
    assert first.preview_handle is not None
    assert second.preview_handle is not None
    assert session._state.revision == revision
    prediction = session.commit(second.preview_handle)
    assert prediction.object_id == 7
    assert prediction.frame_index == 0
    assert prediction.mask.shape == (10, 20)
    assert session._state.revision == revision
    with pytest.raises(PreviewHandleError, match="stale"):
        session.commit(first.preview_handle)
    with pytest.raises(PreviewHandleError, match="already"):
        session.commit(second.preview_handle)
    session.close()


def test_add_remove_readd_uses_minimum_free_slot_without_compaction(
    bundle: Path,
) -> None:
    session = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle
    )
    session.set_video(_frames())
    for object_id in range(17):
        session.add_object(object_id)
    before = session._state.require_object(16)
    session.remove_object(3)
    session.add_object(99)
    assert session._state.require_object(99) == SlotAssignment(0, 3)
    assert session._state.require_object(16) == before == SlotAssignment(1, 0)
    assert session._adapter.counters["state_demuxes"] == 0
    assert session._adapter.counters["state_remuxes"] == 0
    session.close()


@pytest.mark.parametrize("count", [17, 32])
def test_two_bucket_propagation_returns_sorted_public_ids_only(
    bundle: Path, count: int
) -> None:
    session = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle
    )
    session.set_video(_frames(3))
    ids = list(reversed(range(count)))
    for object_id in ids:
        session.add_object(object_id)
        handle = session.preview(
            object_id, 0, options=_single_options()
        ).preview_handle
        assert handle is not None
        session.commit(handle)
    assignment_revision = session._state.revision
    output = session.propagate(start_frame=1, end_frame=2)
    assert [value.frame_index for value in output] == [1, 2]
    assert all(value.object_ids.tolist() == sorted(ids) for value in output)
    assert all(value.masks.shape == (count, 10, 20) for value in output)
    assert session._state.revision == assignment_revision
    assert session._adapter.counters["propagation_launches"] == 2
    assert session._adapter.counters["final_demuxes"] == 2
    assert session._adapter.counters["state_d2h_bytes"] == 0
    assert session._adapter.counters["state_demuxes"] == 0
    assert session._adapter.counters["state_remuxes"] == 0
    session.close()


def test_assignment_update_stales_preview_and_capacity_is_32(bundle: Path) -> None:
    session = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle
    )
    session.set_video(_frames())
    session.add_object(0)
    handle = session.preview(0, 0, options=_single_options()).preview_handle
    assert handle is not None
    session.add_object(1)
    with pytest.raises(PreviewHandleError, match="assignment"):
        session.commit(handle)
    for object_id in range(2, 32):
        session.add_object(object_id)
    with pytest.raises(ObjectStateError, match="capacity is 32"):
        session.add_object(32)
    session.close()


def test_errors_are_deterministic_and_scope_is_separate(bundle: Path) -> None:
    with pytest.raises(PlanNotFoundError, match="unknown multiplex"):
        create_multiplex_video_session("unknown", bundle_dir=bundle)
    with pytest.raises(ManifestError, match="scope mismatch"):
        create_multiplex_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    with pytest.raises(ManifestError, match="scope mismatch"):
        create_video_session(MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle)
    session = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle
    )
    with pytest.raises(VideoStateError, match="set_video"):
        session.add_object(1)
    session.set_video(_frames())
    session.add_object(1)
    with pytest.raises(ObjectStateError, match="duplicate"):
        session.add_object(1)
    with pytest.raises(ObjectStateError, match="unknown"):
        session.remove_object(2)
    with pytest.raises(IndexError, match="outside"):
        session.preview(1, 8)
    with pytest.raises(ValueError):
        session.preview(
            1,
            0,
            InteractivePrompt(
                points_xy=np.asarray([[99.0, 0.0]], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int64),
            ),
        )
    with pytest.raises(ObjectStateError, match="conditioning"):
        session.propagate(start_frame=0, end_frame=1)
    session.close()
    with pytest.raises(SessionClosedError, match="closed"):
        session.close()


def test_foreign_handle_and_public_types_hide_backend_abi(bundle: Path) -> None:
    first = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle
    )
    second = create_multiplex_video_session(
        MULTIPLEX_VIDEO_PLAN_ID, bundle_dir=bundle
    )
    for session in (first, second):
        session.set_video(_frames())
        session.add_object(1)
    foreign = second.preview(1, 0, options=_single_options()).preview_handle
    assert foreign is not None
    with pytest.raises(PreviewHandleError, match="another session"):
        first.commit(foreign)
    public_types = (
        MultiplexVideoSession,
        VideoPreview,
        VideoPrediction,
        VideoFramePrediction,
        type(create_multiplex_video_session),
    )
    annotation_text = " ".join(
        str(typing.get_type_hints(value))
        for value in public_types
        if callable(value)
    )
    public_text = repr(
        (
            first,
            first._video_handle,
            first.preview(1, 0, options=_single_options()),
        )
    )
    for private_term in (
        "OrtValue",
        "bucket_values",
        "object_pointer",
        "memory_features",
        "SlotAssignment",
        "slot_validity",
    ):
        assert private_term not in annotation_text
        assert private_term not in public_text
    first.close()
    second.close()
