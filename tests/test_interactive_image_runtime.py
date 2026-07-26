"""M3 interactive image Public API and host ABI contract tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import sam3
import sam3.export as export_api
from sam3.export.fixtures import InteractiveDecode, PromptEncode
from sam3.runtime import (
    CapabilityError,
    InteractivePredictOptions,
    InteractivePrompt,
    ManifestError,
    PlanNotFoundError,
    SessionClosedError,
    SessionStateError,
    create_image_session,
    create_interactive_session,
    interactive_image,
)
from sam3.runtime.interactive_image import _prompt_arrays
from sam3.runtime.manifest import (
    DEFAULT_PLAN_ID,
    INTERACTIVE_PLAN_ID,
    sha256_file,
    validate_manifest_package,
)

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "tests" / "fixtures" / "manifest_v2" / "minimal_valid.json"


class FakeInteractiveAdapter:
    def __init__(self, _plan: object) -> None:
        self.counters = {
            "image_encodes": 0,
            "predict_launches": 0,
            "session_launches": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "memory_encodes": 0,
            "memory_commits": 0,
        }
        self.last_prompt: dict[str, np.ndarray] | None = None
        self.closed = False

    def encode_image(self, values: np.ndarray) -> object:
        self.counters["image_encodes"] += 1
        return values.shape

    def predict(
        self, image_cache: object, prompt_inputs: dict[str, np.ndarray], multimask: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        del image_cache
        self.last_prompt = deepcopy(prompt_inputs)
        self.counters["predict_launches"] += 1
        self.counters["session_launches"] += 1
        count = 3 if multimask else 1
        logits = np.zeros((count, 288, 288), dtype=np.float32)
        logits[:, :, :144] = 1.0
        scores = np.linspace(0.2, 0.8, count, dtype=np.float32)
        return logits, scores

    def close(self) -> None:
        self.closed = True


def _write_bundle(path: Path) -> Path:
    manifest = json.loads(SKELETON.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        target = path / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(record["id"].encode())
        record["size_bytes"] = target.stat().st_size
        record["digest"]["value"] = sha256_file(target)
    manifest["manifest_id"] = f"{INTERACTIVE_PLAN_ID}-manifest-v2"
    manifest["scope"].update(
        {
            "dispatch_role": "default",
            "scope_label": (
                "SAM3 base interactive image PVS / point-box-mask / ORT CUDA v1"
            ),
            "use_case": "interactive-image-pvs",
            "prompt_coverage": ["point", "box", "mask"],
            "capabilities": ["interactive image PVS", "repeated click"],
            "exclusions": ["video", "memory state", "SAM3.1"],
        }
    )
    manifest["plan"].update(
        {
            "id": INTERACTIVE_PLAN_ID,
            "semantic_graph_kind": "sam3-base-interactive-image-pvs",
            "components": [
                "InteractiveImageEncodeInitial",
                "InteractivePredictMultimask3",
                "InteractivePredictSingle1",
            ],
        }
    )
    manifest["profile"]["id"] = "b1-1008-p16-box1-mask288-fp16"
    manifest["policies"].append(
        {
            "name": "image-condition",
            "binding": "baked",
            "value": "initial-no-memory",
            "stage": "image-cache",
        }
    )
    manifest_dir = path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / f"{INTERACTIVE_PLAN_ID}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    result = _write_bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        interactive_image, "_interactive_adapter_factory", FakeInteractiveAdapter
    )
    return result


def test_prompt_packing_capacity_box_sentinel_and_mask() -> None:
    points = np.asarray([[10.0, 5.0], [50.0, 25.0]], dtype=np.float32)
    mask = np.ones((288, 288), dtype=np.float32)
    packed, facts = _prompt_arrays(
        InteractivePrompt(
            points_xy=points,
            point_labels=np.asarray([1, 0]),
            box_xyxy=(5.0, 2.0, 75.0, 40.0),
            mask_logits=mask,
        ),
        (50, 100),
    )
    np.testing.assert_allclose(
        packed["point-coords"][0, :2], [[100.8, 100.8], [504.0, 504.0]]
    )
    assert packed["point-labels"][0, :3].tolist() == [1, 0, -1]
    assert packed["point-valid"][0].sum() == 2
    np.testing.assert_allclose(packed["box-xyxy"][0], [50.4, 40.32, 756.0, 806.4])
    assert packed["has-box"].tolist() == [True]
    assert packed["has-mask"].tolist() == [True]
    assert packed["mask-input"].shape == (1, 1, 288, 288)
    assert facts == {"point_count": 2, "box_count": 1, "has_mask": True}


@pytest.mark.parametrize("count", [0, 1, 16])
def test_prompt_point_capacities(count: int) -> None:
    prompt = InteractivePrompt(
        points_xy=np.zeros((count, 2), dtype=np.float32),
        point_labels=np.zeros((count,), dtype=np.int64),
    )
    packed, facts = _prompt_arrays(prompt, (10, 10))
    assert packed["point-valid"].sum() == count
    assert facts["point_count"] == count


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        (
            InteractivePrompt(
                points_xy=np.zeros((17, 2)), point_labels=np.zeros((17,))
            ),
            "capacity",
        ),
        (
            InteractivePrompt(points_xy=np.zeros((1, 2)), point_labels=np.asarray([2])),
            "only 0 or 1",
        ),
        (
            InteractivePrompt(points_xy=np.zeros((2, 2)), point_labels=np.asarray([1])),
            "shape must match",
        ),
        (InteractivePrompt(box_xyxy=(5.0, 5.0, 1.0, 8.0)), "x0 < x1"),
        (InteractivePrompt(mask_logits=np.zeros((64, 64))), "mask_logits"),
        (InteractivePrompt(mask_logits=np.zeros((288, 288), dtype=bool)), "numeric"),
    ],
)
def test_invalid_prompts_are_rejected(prompt: InteractivePrompt, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prompt_arrays(prompt, (10, 10))


def test_repeated_click_reuses_image_and_selects_static_artifacts(bundle: Path) -> None:
    session = create_interactive_session(INTERACTIVE_PLAN_ID, bundle_dir=bundle)
    image = Image.new("RGB", (20, 10), "red")
    first_handle = session.set_image(image)
    assert session.set_image(image) == first_handle
    first = session.predict(
        InteractivePrompt(
            points_xy=np.asarray([[5.0, 4.0]]), point_labels=np.asarray([1])
        )
    )
    assert first.low_res_logits.shape == (3, 288, 288)
    selected = first.low_res_logits[int(np.argmax(first.scores))]
    second = session.predict(
        InteractivePrompt(
            points_xy=np.asarray([[5.0, 4.0], [7.0, 5.0]]),
            point_labels=np.asarray([1, 1]),
            mask_logits=selected,
        ),
        InteractivePredictOptions(multimask_output=False, output_size=(6, 8)),
    )
    assert second.masks.shape == (1, 6, 8)
    assert second.metadata["multimask_artifact"] == "single1"
    assert session._adapter.counters["image_encodes"] == 1
    assert session._adapter.counters["predict_launches"] == 2
    assert session._adapter.counters["memory_encodes"] == 0
    assert session._adapter.counters["memory_commits"] == 0
    session.close()


def test_threshold_is_strict_and_image_change_invalidates_cache(bundle: Path) -> None:
    session = create_interactive_session(INTERACTIVE_PLAN_ID, bundle_dir=bundle)
    first = session.set_image(Image.new("RGB", (4, 4), "black"))
    prediction = session.predict(
        options=InteractivePredictOptions(mask_threshold=0.0, output_size=(2, 2))
    )
    assert prediction.masks[:, :, 0].all()
    assert not prediction.masks[:, :, 1].any()
    second = session.set_image(Image.new("RGB", (4, 4), "white"))
    assert first.cache_key != second.cache_key
    assert session._adapter.counters["image_encodes"] == 2
    session.close()


def test_state_close_unknown_and_scope_errors(bundle: Path) -> None:
    with pytest.raises(PlanNotFoundError, match="unknown interactive"):
        create_interactive_session("unknown", bundle_dir=bundle)
    with pytest.raises(ManifestError, match="scope mismatch"):
        create_interactive_session(DEFAULT_PLAN_ID, bundle_dir=bundle)
    with pytest.raises(ManifestError, match="scope mismatch"):
        create_image_session(INTERACTIVE_PLAN_ID, bundle_dir=bundle)

    session = create_interactive_session(INTERACTIVE_PLAN_ID, bundle_dir=bundle)
    with pytest.raises(SessionStateError, match="set_image"):
        session.predict()
    session.close()
    with pytest.raises(SessionClosedError, match="closed"):
        session.close()
    with pytest.raises(SessionClosedError, match="closed"):
        session.set_image(Image.new("RGB", (2, 2)))


def test_interactive_package_hash_file_and_capability_are_validated(
    tmp_path: Path,
) -> None:
    tampered = _write_bundle(tmp_path / "tampered")
    manifest_path = tampered / "manifests" / f"{INTERACTIVE_PLAN_ID}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["digest"]["value"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="hash mismatch"):
        validate_manifest_package(manifest_path)

    missing = _write_bundle(tmp_path / "missing")
    missing_path = missing / "manifests" / f"{INTERACTIVE_PLAN_ID}.json"
    missing_manifest = json.loads(missing_path.read_text(encoding="utf-8"))
    (missing / missing_manifest["files"][0]["path"]).unlink()
    with pytest.raises(ManifestError, match="file is missing"):
        validate_manifest_package(missing_path)

    capability = _write_bundle(tmp_path / "capability")
    capability_path = capability / "manifests" / f"{INTERACTIVE_PLAN_ID}.json"
    capability_manifest = json.loads(capability_path.read_text(encoding="utf-8"))
    capability_manifest["backend"]["capabilities"].remove("iobinding")
    capability_path.write_text(json.dumps(capability_manifest), encoding="utf-8")
    with pytest.raises(CapabilityError, match="required capabilities"):
        validate_manifest_package(capability_path)


def test_tiny_interactive_wrappers_are_fixture_namespace_only() -> None:
    assert PromptEncode.__module__.startswith("sam3.export.fixtures.")
    assert InteractiveDecode.__module__.startswith("sam3.export.fixtures.")
    assert not hasattr(export_api, "PromptEncode")
    assert not hasattr(export_api, "InteractiveDecode")
    assert not hasattr(sam3, "PromptEncode")
    assert not hasattr(sam3, "InteractiveDecode")


def test_public_values_do_not_expose_backend_abi(bundle: Path) -> None:
    session = create_interactive_session(INTERACTIVE_PLAN_ID, bundle_dir=bundle)
    session.set_image(Image.new("RGB", (3, 2)))
    prediction = session.predict()
    public_text = repr(
        (session, InteractivePrompt(), InteractivePredictOptions(), prediction)
    )
    assert "OrtValue" not in public_text
    assert "image_embedding" not in public_text
    assert "session slot" not in public_text
    session.close()
