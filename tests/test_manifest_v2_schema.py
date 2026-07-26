"""Structural gates for the draft deployment manifest v2 contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "sam3-deployment-manifest-v2.schema.json"
VALID_FIXTURE_PATH = ROOT / "tests" / "fixtures" / "manifest_v2" / "minimal_valid.json"


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_manifest_v2_schema_and_valid_fixture() -> None:
    schema = _load(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_load(VALID_FIXTURE_PATH))


def test_manifest_v2_requires_checkpoint_digest() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    del manifest["model"]["checkpoint"]["digest"]  # type: ignore[index]

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "required" for error in errors)


def test_manifest_v2_rejects_bounded_profile_without_ranges() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    manifest["profile"]["shape_mode"] = "bounded-dynamic"  # type: ignore[index]

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "minItems" for error in errors)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/absolute.onnx",
        "../outside.onnx",
        "graphs/../../outside.onnx",
        "graphs/./model.onnx",
        "graphs//model.onnx",
        "graphs/",
    ],
)
def test_manifest_v2_rejects_unsafe_package_paths(unsafe_path: str) -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    manifest["files"][0]["path"] = unsafe_path  # type: ignore[index]

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "pattern" for error in errors)


def test_manifest_v2_requires_all_four_distinct_parity_stages() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    parity = manifest["fixtures"][0]["parity"]  # type: ignore[index]
    parity[1]["stage"] = "official-to-local-eager"

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator in {"contains", "maxContains"} for error in errors)


def test_manifest_v2_requires_report_for_completed_parity() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    parity = manifest["fixtures"][0]["parity"]  # type: ignore[index]
    parity[0]["status"] = "pass"

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "type" for error in errors)


def test_manifest_v2_requires_report_for_completed_strict_audit() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    strict_audit = manifest["capture"]["strict_audit"]  # type: ignore[index]
    strict_audit["status"] = "pass"

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "type" for error in errors)


def test_manifest_v2_requires_integer_opset_for_onnx_runtime() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    backend = manifest["backend"]  # type: ignore[assignment]
    backend["kind"] = "onnx-runtime"

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "type" for error in errors)


def test_manifest_v2_rejects_fixture_with_public_dispatch_role() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    manifest["scope"]["dispatch_role"] = "default"  # type: ignore[index]

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "const" for error in errors)


def test_manifest_v2_requires_dispatch_role_for_shipped_public_plan() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    scope = manifest["scope"]  # type: ignore[assignment]
    scope["classification"] = "public-deployment"
    scope["lifecycle"] = "shipped"
    scope["dispatch_role"] = "not-applicable"

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "enum" for error in errors)


@pytest.mark.parametrize(
    ("canonical_format", "mode"),
    [
        ("exported-program", "synthetic"),
        ("synthetic", "strict"),
        ("synthetic", "non-strict"),
    ],
)
def test_manifest_v2_rejects_incompatible_capture_mode(
    canonical_format: str, mode: str
) -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    capture = manifest["capture"]  # type: ignore[assignment]
    capture["canonical_format"] = canonical_format
    capture["mode"] = mode

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator in {"enum", "const"} for error in errors)
