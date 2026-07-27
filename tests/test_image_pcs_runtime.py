"""M2 image PCS manifest and Public API contract tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from sam3.grounding.tokenizer_ve import DEFAULT_BPE_PATH
from sam3.runtime import (
    CapabilityError,
    LegacyManifestError,
    ManifestError,
    PlanNotFoundError,
    PredictOptions,
    SessionClosedError,
    SessionStateError,
    create_image_session,
    image_pcs,
)
from sam3.runtime.image_pcs import _BackendPrediction
from sam3.runtime.manifest import (
    AOTINDUCTOR_PLAN_ID,
    DEFAULT_PLAN_ID,
    EXPORTED_PROGRAM_PLAN_ID,
    SELECTED_K32_PLAN_ID,
    SPLIT_PLAN_ID,
    manifest_format,
    sha256_file,
    validate_manifest_package,
)

ROOT = Path(__file__).resolve().parents[1]
SKELETON_PATH = ROOT / "tests" / "fixtures" / "manifest_v2" / "minimal_valid.json"


class FakeAdapter:
    def __init__(self, _plan: object) -> None:
        self.counters = {
            "image_encodes": 0,
            "text_encodes": 0,
            "session_launches": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "mask_skips": 0,
        }
        self.closed = False

    def encode_image(self, values: np.ndarray) -> object:
        self.counters["image_encodes"] += 1
        return values.shape

    def encode_text(self, token_ids: np.ndarray, attention_mask: np.ndarray) -> object:
        self.counters["text_encodes"] += 1
        return (token_ids.copy(), attention_mask.copy())

    def predict(
        self, image_cache: object, text_cache: object, score_threshold: float
    ) -> _BackendPrediction:
        del image_cache, text_cache, score_threshold
        self.counters["session_launches"] += 1
        masks = np.zeros((3, 2, 2), dtype=np.float32)
        masks[0] = 2.0
        masks[1] = np.asarray([[2.0, 2.0], [2.0, -2.0]])
        masks[2] = masks[1]
        return _BackendPrediction(
            boxes_cxcywh=np.asarray(
                [[0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]],
                dtype=np.float32,
            ),
            scores=np.asarray([0.5, 0.9, 0.9], dtype=np.float32),
            mask_logits=masks,
            query_indices=np.asarray([0, 2, 1], dtype=np.int64),
        )

    def close(self) -> None:
        self.closed = True


def _write_bundle(path: Path) -> Path:
    skeleton = json.loads(SKELETON_PATH.read_text(encoding="utf-8"))
    tokenizer_path = path / "tokenizer" / DEFAULT_BPE_PATH.name
    tokenizer_path.parent.mkdir(parents=True)
    shutil.copy2(DEFAULT_BPE_PATH, tokenizer_path)

    for record in skeleton["files"]:
        target = path / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(record["id"].encode())
        record["size_bytes"] = target.stat().st_size
        record["digest"]["value"] = sha256_file(target)
    skeleton["files"].append(
        {
            "id": "tokenizer-bpe",
            "path": f"tokenizer/{DEFAULT_BPE_PATH.name}",
            "role": "tokenizer",
            "size_bytes": tokenizer_path.stat().st_size,
            "digest": {"algorithm": "sha256", "value": sha256_file(tokenizer_path)},
        }
    )
    skeleton["caches"].append(
        {
            "id": "prompt-cache",
            "tensor_refs": ["encoded-values"],
            "lifetime": "prompt-session",
            "key_version": "1.0.0",
            "key_parts": ["token-ids", "tokenizer-digest", "model-revision"],
            "invalidated_by": ["text-change", "tokenizer-change"],
            "state_compatibility": "matching tokenizer, checkpoint, and profile",
        }
    )

    manifest_dir = path / "manifests"
    manifest_dir.mkdir()
    for plan_id, role in (
        (DEFAULT_PLAN_ID, "default"),
        (SELECTED_K32_PLAN_ID, "optional"),
        (SPLIT_PLAN_ID, "fallback"),
    ):
        manifest = deepcopy(skeleton)
        manifest["manifest_id"] = f"{plan_id}-manifest-v2"
        manifest["plan"]["id"] = plan_id
        manifest["scope"]["dispatch_role"] = role
        fallback = DEFAULT_PLAN_ID if plan_id == SELECTED_K32_PLAN_ID else None
        for handoff in manifest["handoffs"]:
            handoff["fallback_plan_id"] = fallback
        (manifest_dir / f"{plan_id}.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return path


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _write_bundle(tmp_path / "bundle")
    monkeypatch.setattr(image_pcs, "_adapter_factory", FakeAdapter)
    return path


def test_manifest_formats_are_dispatched_without_v1_inference(tmp_path: Path) -> None:
    v1 = tmp_path / "manifest.json"
    v1.write_text('{"format":"sam3-split-onnx-v1"}\n', encoding="utf-8")
    assert manifest_format(v1) == "sam3-split-onnx-v1"
    with pytest.raises(LegacyManifestError, match="separate legacy format"):
        validate_manifest_package(v1)


def test_unknown_plan_is_explicit(bundle: Path) -> None:
    with pytest.raises(PlanNotFoundError, match="unknown image PCS plan"):
        create_image_session("unknown-plan", bundle_dir=bundle)


def test_package_hash_and_required_capability_are_validated(bundle: Path) -> None:
    manifest_path = bundle / "manifests" / f"{DEFAULT_PLAN_ID}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["digest"]["value"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="hash mismatch"):
        validate_manifest_package(manifest_path)

    _write_bundle(bundle.parent / "capability-bundle")
    capability_path = (
        bundle.parent / "capability-bundle" / "manifests" / f"{DEFAULT_PLAN_ID}.json"
    )
    manifest = json.loads(capability_path.read_text(encoding="utf-8"))
    manifest["backend"]["capabilities"].remove("iobinding")
    capability_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CapabilityError, match="required capabilities"):
        validate_manifest_package(capability_path)

    missing_bundle = _write_bundle(bundle.parent / "missing-file-bundle")
    missing_manifest = missing_bundle / "manifests" / f"{DEFAULT_PLAN_ID}.json"
    missing_record = json.loads(missing_manifest.read_text(encoding="utf-8"))["files"][
        0
    ]
    (missing_bundle / missing_record["path"]).unlink()
    with pytest.raises(ManifestError, match="file is missing"):
        validate_manifest_package(missing_manifest)


def test_aotinductor_candidate_uses_the_same_public_api(bundle: Path) -> None:
    source_path = bundle / "manifests" / f"{DEFAULT_PLAN_ID}.json"
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    manifest["manifest_id"] = "aotinductor-evaluation-manifest-v2"
    manifest["plan"]["id"] = AOTINDUCTOR_PLAN_ID
    manifest["scope"]["lifecycle"] = "candidate"
    manifest["scope"]["dispatch_role"] = "optional"
    manifest["backend"] = {
        "kind": "aotinductor",
        "target": "CUDA device 0",
        "execution_provider": "TorchInductorCUDA",
        "runtime_version": "2.13.0",
        "pytorch_version": "2.13.0",
        "exporter_version": "2.13.0",
        "opset": None,
        "capabilities": ["device-resident-handoff"],
    }
    manifest_path = bundle / "manifests" / f"{AOTINDUCTOR_PLAN_ID}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved = validate_manifest_package(
        manifest_path, expected_plan_id=AOTINDUCTOR_PLAN_ID
    )
    assert resolved.manifest["backend"]["kind"] == "aotinductor"

    session = create_image_session(AOTINDUCTOR_PLAN_ID, bundle_dir=bundle)
    session.set_image(Image.new("RGB", (8, 6), "red"))
    session.set_text("a wheel")
    prediction = session.predict_text()
    assert prediction.metadata["plan_id"] == AOTINDUCTOR_PLAN_ID
    session.close()


def test_exported_program_candidate_uses_the_same_public_api(bundle: Path) -> None:
    source_path = bundle / "manifests" / f"{DEFAULT_PLAN_ID}.json"
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    manifest["manifest_id"] = "exported-program-evaluation-manifest-v2"
    manifest["plan"]["id"] = EXPORTED_PROGRAM_PLAN_ID
    manifest["scope"]["lifecycle"] = "candidate"
    manifest["scope"]["dispatch_role"] = "optional"
    manifest["backend"] = {
        "kind": "exported-program",
        "target": "CUDA device 0",
        "execution_provider": "PyTorchATenCUDA",
        "runtime_version": "2.13.0",
        "pytorch_version": "2.13.0",
        "exporter_version": "2.13.0",
        "opset": None,
        "capabilities": ["device-resident-handoff"],
    }
    manifest_path = bundle / "manifests" / f"{EXPORTED_PROGRAM_PLAN_ID}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved = validate_manifest_package(
        manifest_path, expected_plan_id=EXPORTED_PROGRAM_PLAN_ID
    )
    assert resolved.manifest["backend"]["kind"] == "exported-program"

    session = create_image_session(EXPORTED_PROGRAM_PLAN_ID, bundle_dir=bundle)
    session.set_image(Image.new("RGB", (8, 6), "red"))
    session.set_text("a wheel")
    prediction = session.predict_text()
    assert prediction.metadata["plan_id"] == EXPORTED_PROGRAM_PLAN_ID
    session.close()


def test_missing_binding_reference_and_fallback_are_rejected(bundle: Path) -> None:
    manifest_path = bundle / "manifests" / f"{SELECTED_K32_PLAN_ID}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["inputs"][0]["tensor_ref"] = "missing-tensor"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="references missing IDs"):
        validate_manifest_package(manifest_path)

    fallback_path = bundle / "manifests" / f"{DEFAULT_PLAN_ID}.json"
    fallback_path.unlink()
    manifest["artifacts"][0]["inputs"][0]["tensor_ref"] = "input-values"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="fallback plan is missing"):
        validate_manifest_package(manifest_path)


def test_image_and_text_caches_reuse_and_invalidate_independently(bundle: Path) -> None:
    session = create_image_session(DEFAULT_PLAN_ID, bundle_dir=bundle)
    image_a = Image.new("RGB", (8, 6), "red")
    image_b = Image.new("RGB", (8, 6), "blue")

    first_image = session.set_image(image_a)
    assert session.set_image(image_a) == first_image
    first_prompt = session.set_text("a wheel")
    assert session.set_text("a wheel") == first_prompt
    assert session._adapter.counters["image_encodes"] == 1
    assert session._adapter.counters["text_encodes"] == 1

    assert session.set_image(image_b).cache_key != first_image.cache_key
    assert session._adapter.counters["image_encodes"] == 2
    assert session._adapter.counters["text_encodes"] == 1
    assert session.set_text("a truck").cache_key != first_prompt.cache_key
    assert session._adapter.counters["image_encodes"] == 2
    assert session._adapter.counters["text_encodes"] == 2
    session.close()


def test_state_and_close_errors_are_deterministic(bundle: Path) -> None:
    session = create_image_session(DEFAULT_PLAN_ID, bundle_dir=bundle)
    with pytest.raises(SessionStateError, match="set_image"):
        session.predict_text()
    session.set_image(Image.new("RGB", (3, 2)))
    with pytest.raises(SessionStateError, match="set_text"):
        session.predict_text()
    session.close()
    with pytest.raises(SessionClosedError, match="closed"):
        session.close()
    with pytest.raises(SessionClosedError, match="closed"):
        session.set_text("a truck")


def test_strict_threshold_tie_order_nms_empty_and_output_size(bundle: Path) -> None:
    session = create_image_session(DEFAULT_PLAN_ID, bundle_dir=bundle)
    session.set_image(Image.new("RGB", (20, 10)))
    session.set_text("object")

    prediction = session.predict_text(
        PredictOptions(score_threshold=0.5, output_size=(6, 8))
    )
    assert session._last_query_indices.tolist() == [1, 2]
    assert prediction.boxes_xyxy.dtype == np.float32
    assert prediction.scores.dtype == np.float32
    assert prediction.masks.shape == (2, 6, 8)
    assert prediction.metadata["original_size"] == (10, 20)
    assert prediction.metadata["output_size"] == (6, 8)

    suppressed = session.predict_text(
        PredictOptions(score_threshold=0.5, nms_iou_threshold=0.5)
    )
    assert suppressed.scores.shape == (1,)
    empty = session.predict_text(PredictOptions(score_threshold=0.99))
    assert empty.boxes_xyxy.shape == (0, 4)
    assert empty.masks.shape == (0, 10, 20)
    session.close()


def test_result_limits_and_selected_k_capacity(bundle: Path) -> None:
    default = create_image_session(DEFAULT_PLAN_ID, bundle_dir=bundle)
    default.set_image(Image.new("RGB", (4, 4)))
    default.set_text("object")
    assert default.predict_text(PredictOptions(max_results=1)).scores.shape == (1,)
    default.close()

    selected = create_image_session(SELECTED_K32_PLAN_ID, bundle_dir=bundle)
    selected.set_image(Image.new("RGB", (4, 4)))
    selected.set_text("object")
    with pytest.raises(ValueError, match="more than 32"):
        selected.predict_text(PredictOptions(max_results=33))
    selected.close()


def test_public_types_do_not_expose_backend_abi(bundle: Path) -> None:
    session = create_image_session(DEFAULT_PLAN_ID, bundle_dir=bundle)
    session.set_image(Image.new("RGB", (4, 4)))
    session.set_text("object")
    prediction = session.predict_text()
    public_text = repr((session, prediction, PredictOptions()))
    assert "OrtValue" not in public_text
    assert "image_feature" not in public_text
    assert "session slot" not in public_text
    session.close()
