"""Structural gates for the M2 deployment manifest v2 release contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

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
    backend["opset"] = None

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "type" for error in errors)


def test_manifest_v2_rejects_non_release_dispatch_role() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    manifest["scope"]["dispatch_role"] = "legacy"  # type: ignore[index]

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "enum" for error in errors)


def test_manifest_v2_requires_dispatch_role_for_shipped_public_plan() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    scope = manifest["scope"]  # type: ignore[assignment]
    scope["classification"] = "public-deployment"
    scope["lifecycle"] = "shipped"
    scope["dispatch_role"] = "not-applicable"

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "enum" for error in errors)


def test_manifest_v2_rejects_incompatible_capture_mode() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    capture = manifest["capture"]  # type: ignore[assignment]
    capture["mode"] = "synthetic"

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "enum" for error in errors)


def test_manifest_v2_requires_canonical_program_file_refs() -> None:
    schema = _load(SCHEMA_PATH)
    manifest = deepcopy(_load(VALID_FIXTURE_PATH))
    del manifest["capture"]["program_file_refs"]  # type: ignore[index]

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(error.validator == "required" for error in errors)
