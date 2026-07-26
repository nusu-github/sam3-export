"""M4 base video Public API and fake-adapter contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from sam3.runtime import (
    BASE_VIDEO_PLAN_ID,
    InteractivePredictOptions,
    InteractivePrompt,
    ManifestError,
    ObjectStateError,
    PlanNotFoundError,
    PreviewHandleError,
    SessionClosedError,
    StateCapacityError,
    VideoStateError,
    base_video,
    create_image_session,
    create_video_session,
)
from sam3.runtime.base_video import _BackendCommit, _BackendPreviewBatch
from sam3.runtime.manifest import DEFAULT_PLAN_ID, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "tests" / "fixtures" / "manifest_v2" / "minimal_valid.json"


VARIANT = {
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


class FakeBaseVideoAdapter:
    def __init__(self, _plan: object) -> None:
        self.batch_capacity = 4
        self.fused_default = True
        self.counters = {
            "frame_encodes": 0,
            "preview_launches": 0,
            "tracker_launches": 0,
            "memory_encodes": 0,
            "memory_commits": 0,
            "session_launches": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "d2d_pack_bytes": 0,
        }
        self.closed = False

    def encode_frame(self, values: np.ndarray) -> object:
        self.counters["frame_encodes"] += 1
        return ("frame", values.shape, self.counters["frame_encodes"])

    def preview(
        self,
        frame_cache: object,
        packed: object,
        prompt_inputs: dict[str, np.ndarray],
        *,
        multimask: bool,
    ) -> _BackendPreviewBatch:
        del frame_cache, prompt_inputs
        self.counters["preview_launches"] += 1
        self.counters["tracker_launches"] += 1
        self.counters["session_launches"] += 1
        count = 3 if multimask else 1
        logits = []
        scores = []
        for object_id in packed.object_ids:
            if object_id is None:
                logits.append(None)
                scores.append(None)
                continue
            value = np.full((count, 288, 288), float(object_id), dtype=np.float32)
            value[:, :, :144] += 1.0
            logits.append(value)
            scores.append(np.linspace(0.2, 0.8, count, dtype=np.float32))
        return _BackendPreviewBatch(
            low_res_logits=tuple(logits),
            scores=tuple(scores),
            commit_masks=("commit-mask", packed.object_ids),
            object_pointers=("pointers", packed.object_ids),
            object_scores=("object-scores", packed.object_ids),
        )

    def commit(
        self,
        frame_cache: object,
        packed: object,
        preview: _BackendPreviewBatch,
        *,
        is_mask_from_points: bool,
    ) -> tuple[_BackendCommit | None, ...]:
        del frame_cache
        self.counters["memory_encodes"] += 1
        self.counters["session_launches"] += 1
        self.counters["memory_commits"] += int(packed.object_valid.sum())
        result = []
        for row, object_id in enumerate(packed.object_ids):
            if object_id is None:
                result.append(None)
                continue
            logits = preview.low_res_logits[row]
            assert logits is not None
            high = np.full(
                (1, 10, 20),
                1.0 if is_mask_from_points else float(object_id) + 1.0,
                dtype=np.float32,
            )
            result.append(
                _BackendCommit(
                    memory_features=("memory", object_id),
                    memory_position=("position", object_id),
                    object_pointer=("pointer", object_id),
                    high_res_logits=high,
                    object_score=1.0,
                )
            )
        return tuple(result)

    def step_and_commit(
        self,
        frame_cache: object,
        packed: object,
        prompt_inputs: dict[str, np.ndarray],
    ) -> tuple[_BackendPreviewBatch, tuple[_BackendCommit | None, ...]]:
        preview = self.preview(frame_cache, packed, prompt_inputs, multimask=False)
        return preview, self.commit(
            frame_cache, packed, preview, is_mask_from_points=False
        )

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
    manifest["manifest_id"] = f"{BASE_VIDEO_PLAN_ID}-manifest-v2"
    manifest["scope"].update(
        {
            "scope_label": (
                "SAM3 base video tracking / point-box-mask correction / "
                "per-object batch / ORT CUDA v1"
            ),
            "use_case": "base-video-tracking",
            "prompt_coverage": ["point", "box", "mask"],
            "capabilities": ["base video tracking", "per-object batch"],
            "exclusions": ["SAM3.1", "Multiplex", "CPU fallback"],
        }
    )
    manifest["plan"].update(
        {
            "id": BASE_VIDEO_PLAN_ID,
            "semantic_graph_kind": "sam3-base-video-tracking",
            "components": ["BaseTrackerPreview", "BaseMemoryCommit"],
        }
    )
    manifest["model"]["tracking_layout"] = "BaseVideoStateV1"
    manifest["model"]["variant_parameters"] = [
        {"name": name, "value": value} for name, value in variant.items()
    ]
    manifest["profile"]["id"] = "b4-1008-p16-box1-mask288-fp16"
    manifest["profile"]["static_values"].append(
        {"name": "object-batch-capacity", "value": 4}
    )
    manifest["caches"][0].update(
        {
            "id": "frame-cache",
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
            "state_compatibility": "memory-aware-frame-view-v1",
        }
    )
    manifest["policies"].append(
        {
            "name": "steady-state-cut",
            "binding": "baked",
            "value": "fused",
            "stage": "propagation",
        }
    )
    manifest_dir = path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / f"{BASE_VIDEO_PLAN_ID}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    result = _write_bundle(tmp_path / "bundle")
    monkeypatch.setattr(base_video, "_base_video_adapter_factory", FakeBaseVideoAdapter)
    return result


def _frames(count: int = 4) -> list[Image.Image]:
    return [Image.new("RGB", (20, 10), (index, 0, 0)) for index in range(count)]


def _single_options() -> InteractivePredictOptions:
    return InteractivePredictOptions(multimask_output=False)


def test_repeated_preview_is_non_mutating_and_only_single_commits(bundle: Path) -> None:
    session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    session.set_video(_frames())
    session.add_object(5)
    prompt = InteractivePrompt(
        points_xy=np.asarray([[4.0, 3.0]], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int64),
    )
    first = session.preview(5, 0, prompt)
    second = session.preview(5, 0, prompt)
    assert first.preview_handle is None
    assert second.preview_handle is None
    assert session._state.revision == 0
    assert session._adapter.counters["memory_commits"] == 0
    with pytest.raises(PreviewHandleError, match="PreviewHandle"):
        session.commit(first.preview_handle)

    final = session.preview(5, 0, prompt, _single_options())
    assert final.preview_handle is not None
    prediction = session.commit(final.preview_handle)
    assert prediction.object_id == 5
    assert prediction.frame_index == 0
    assert prediction.mask.shape == (10, 20)
    assert session._state.revision == 1
    assert session._adapter.counters["memory_commits"] == 1
    with pytest.raises(PreviewHandleError, match="already"):
        session.commit(final.preview_handle)
    session.close()


def test_stale_foreign_and_scope_handles(bundle: Path) -> None:
    first = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    second = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    for session in (first, second):
        session.set_video(_frames())
        session.add_object(1)
    stale = first.preview(1, 0, options=_single_options()).preview_handle
    current = first.preview(1, 0, options=_single_options()).preview_handle
    assert stale is not None and current is not None
    first.commit(current)
    with pytest.raises(PreviewHandleError, match="stale"):
        first.commit(stale)
    foreign = second.preview(1, 0, options=_single_options()).preview_handle
    assert foreign is not None
    with pytest.raises(PreviewHandleError, match="another session"):
        first.commit(foreign)
    first.close()
    second.close()


def test_propagate_chunks_objects_and_encodes_each_frame_once(bundle: Path) -> None:
    session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    session.set_video(_frames(3))
    for object_id in range(5):
        session.add_object(object_id)
        handle = session.preview(object_id, 0, options=_single_options()).preview_handle
        assert handle is not None
        session.commit(handle)
    before = dict(session._adapter.counters)
    output = session.propagate(start_frame=0, end_frame=2)
    assert [value.frame_index for value in output] == [0, 1, 2]
    assert all(value.object_ids.tolist() == [0, 1, 2, 3, 4] for value in output)
    assert all(value.masks.shape == (5, 10, 20) for value in output)
    assert session._adapter.counters["frame_encodes"] - before["frame_encodes"] == 2
    assert (
        session._adapter.counters["tracker_launches"] - before["tracker_launches"] == 4
    )
    assert session._adapter.counters["memory_commits"] - before["memory_commits"] == 10
    session.close()


def test_correction_replacement_invalidates_non_conditioning(bundle: Path) -> None:
    session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    session.set_video(_frames(4))
    session.add_object(9)
    initial = session.preview(9, 0, options=_single_options()).preview_handle
    assert initial is not None
    session.commit(initial)
    session.propagate(start_frame=1, end_frame=3)
    assert list(session._state.require_object(9).non_conditioning) == [1, 2, 3]
    correction = session.preview(9, 2, options=_single_options()).preview_handle
    assert correction is not None
    session.commit(correction)
    state = session._state.require_object(9)
    assert list(state.non_conditioning) == [1]
    assert 2 in state.conditioning
    session.close()


def test_conditioning_capacity_has_no_silent_eviction(bundle: Path) -> None:
    session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    session.set_video(_frames(5))
    session.add_object(1)
    for frame_index in range(4):
        handle = session.preview(
            1, frame_index, options=_single_options()
        ).preview_handle
        assert handle is not None
        session.commit(handle)
    replacement = session.preview(1, 3, options=_single_options()).preview_handle
    assert replacement is not None
    session.commit(replacement)
    extra = session.preview(1, 4, options=_single_options()).preview_handle
    assert extra is not None
    with pytest.raises(StateCapacityError, match="capacity"):
        session.commit(extra)
    assert list(session._state.require_object(1).conditioning) == [0, 1, 2, 3]
    session.close()


def test_video_object_range_scope_and_close_errors(bundle: Path) -> None:
    with pytest.raises(PlanNotFoundError, match="unknown base video"):
        create_video_session("unknown", bundle_dir=bundle)
    with pytest.raises(ManifestError, match="scope mismatch"):
        create_video_session(DEFAULT_PLAN_ID, bundle_dir=bundle)
    with pytest.raises(ManifestError, match="scope mismatch"):
        create_image_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)

    session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    with pytest.raises(VideoStateError, match="set_video"):
        session.add_object(1)
    with pytest.raises(ValueError, match="non-empty"):
        session.set_video([])
    with pytest.raises(ValueError, match="identical"):
        session.set_video([Image.new("RGB", (2, 2)), Image.new("RGB", (3, 2))])
    session.set_video(_frames())
    session.add_object(1)
    with pytest.raises(ObjectStateError, match="duplicate"):
        session.add_object(1)
    with pytest.raises(ObjectStateError, match="unknown"):
        session.preview(2, 0)
    with pytest.raises(IndexError, match="outside"):
        session.preview(1, 8)
    with pytest.raises(ObjectStateError, match="conditioning"):
        session.propagate(start_frame=0, end_frame=1)
    with pytest.raises(ValueError, match="direction"):
        session.propagate(start_frame=2, end_frame=1, reverse=False)
    session.close()
    with pytest.raises(SessionClosedError, match="closed"):
        session.close()


def test_public_values_do_not_expose_backend_state(bundle: Path) -> None:
    session = create_video_session(BASE_VIDEO_PLAN_ID, bundle_dir=bundle)
    video = session.set_video(_frames())
    session.add_object(1)
    preview = session.preview(1, 0, options=_single_options())
    public_text = repr((session, video, preview, preview.preview_handle))
    for private_term in (
        "OrtValue",
        "memory_features",
        "frame_image_embedding",
        "slot",
    ):
        assert private_term not in public_text
    session.close()
