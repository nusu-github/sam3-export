"""Deployment manifest loading and current release package validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

MANIFEST_FORMAT_V1 = "sam3-split-onnx-v1"
MANIFEST_FORMAT_V2 = "sam3-deployment-manifest-v2"
DEFAULT_PLAN_ID = "sam3_base_image_pcs_text_ortcuda_v1"
SELECTED_K32_PLAN_ID = "sam3_base_image_pcs_text_ortcuda_selected_k32_v1"
SPLIT_PLAN_ID = "sam3_base_image_pcs_text_ortcuda_split_v1"
INTERACTIVE_PLAN_ID = "sam3_base_interactive_image_pvs_ortcuda_v1"
BASE_VIDEO_PLAN_ID = "sam3_base_video_tracking_ortcuda_v1"
IMAGE_PCS_PLAN_IDS = frozenset({DEFAULT_PLAN_ID, SELECTED_K32_PLAN_ID, SPLIT_PLAN_ID})
INTERACTIVE_PLAN_IDS = frozenset({INTERACTIVE_PLAN_ID})
BASE_VIDEO_PLAN_IDS = frozenset({BASE_VIDEO_PLAN_ID})
SUPPORTED_PLAN_IDS = IMAGE_PCS_PLAN_IDS | INTERACTIVE_PLAN_IDS | BASE_VIDEO_PLAN_IDS


class ManifestError(RuntimeError):
    """The requested deployment manifest or package is invalid."""


class PlanNotFoundError(ManifestError):
    """The requested public plan does not exist in the bundle."""


class LegacyManifestError(ManifestError):
    """A v1 manifest was supplied where a v2 deployment plan is required."""


class CapabilityError(RuntimeError):
    """The runtime cannot meet a required deployment capability."""


@dataclass(frozen=True)
class ResolvedPlan:
    """Validated runtime view of one shipped deployment plan."""

    bundle_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_digest: str

    @property
    def plan_id(self) -> str:
        return str(self.manifest["plan"]["id"])

    @property
    def contract_version(self) -> str:
        return str(self.manifest["plan"]["contract_version"])

    @property
    def profile_id(self) -> str:
        return str(self.manifest["profile"]["id"])

    @property
    def dispatch_role(self) -> str:
        return str(self.manifest["scope"]["dispatch_role"])

    @property
    def artifacts_by_role(self) -> dict[str, dict[str, Any]]:
        return {str(item["role"]): item for item in self.manifest["artifacts"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_path() -> Path:
    source_tree = Path(__file__).resolve().parents[3] / "schemas"
    schema = source_tree / "sam3-deployment-manifest-v2.schema.json"
    if schema.is_file():
        return schema

    # Wheels install the schema as a data file under share/sam3/schemas.
    import sysconfig

    installed = (
        Path(sysconfig.get_path("data")) / "share" / "sam3" / "schemas" / schema.name
    )
    if installed.is_file():
        return installed
    raise ManifestError("deployment manifest v2 schema is not installed")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest must be a JSON object: {path}")
    return value


def manifest_format(path: str | Path) -> str:
    """Read only the format discriminator; v1 is never interpreted as v2."""

    value = _load_json(Path(path))
    format_name = value.get("format")
    if not isinstance(format_name, str):
        raise ManifestError("manifest format is missing")
    return format_name


def _unique_map(items: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item["id"])
        if item_id in result:
            raise ManifestError(f"duplicate {label} id: {item_id}")
        result[item_id] = item
    return result


def _require_refs(refs: list[str], available: dict[str, object], context: str) -> None:
    missing = sorted(set(refs) - set(available))
    if missing:
        raise ManifestError(f"{context} references missing IDs: {missing}")


def _validate_references(manifest: dict[str, Any], bundle_dir: Path) -> None:
    tensors = _unique_map(manifest["tensors"], "tensor")
    artifacts = _unique_map(manifest["artifacts"], "artifact")
    files = _unique_map(manifest["files"], "file")

    for artifact in artifacts.values():
        _require_refs(
            [artifact["entry_file_ref"], *artifact["external_data_file_refs"]],
            files,
            f"artifact {artifact['id']} files",
        )
        _require_refs(
            [binding["tensor_ref"] for binding in artifact["inputs"]],
            tensors,
            f"artifact {artifact['id']} inputs",
        )
        _require_refs(
            [binding["tensor_ref"] for binding in artifact["outputs"]],
            tensors,
            f"artifact {artifact['id']} outputs",
        )

    execution = manifest["execution"]
    _require_refs(execution["entry_artifacts"], artifacts, "execution entries")
    for edge in execution["edges"]:
        _require_refs(
            [edge["producer_artifact_ref"], edge["consumer_artifact_ref"]],
            artifacts,
            "execution edge artifacts",
        )
        _require_refs(edge["tensor_refs"], tensors, "execution edge tensors")

    for cache in manifest["caches"]:
        _require_refs(cache["tensor_refs"], tensors, f"cache {cache['id']}")
    for handoff in manifest["handoffs"]:
        _require_refs(
            [handoff["producer_artifact_ref"], handoff["consumer_artifact_ref"]],
            artifacts,
            f"handoff {handoff['id']} artifacts",
        )
        _require_refs(
            handoff["tensor_refs"], tensors, f"handoff {handoff['id']} tensors"
        )
        fallback = handoff["fallback_plan_id"]
        if fallback is not None:
            fallback_path = bundle_dir / "manifests" / f"{fallback}.json"
            if not fallback_path.is_file():
                raise ManifestError(
                    f"handoff {handoff['id']} fallback plan is missing: {fallback}"
                )

    capture_ref = manifest["capture"]["graph_signature_file_ref"]
    _require_refs([capture_ref], files, "capture graph signature")
    for fixture in manifest["fixtures"]:
        report_refs = [
            item["report_file_ref"]
            for item in fixture["parity"]
            if item["report_file_ref"] is not None
        ]
        _require_refs(report_refs, files, f"fixture {fixture['id']} reports")


def _validate_files(manifest: dict[str, Any], bundle_dir: Path) -> None:
    root = bundle_dir.resolve()
    for record in manifest["files"]:
        candidate = (root / record["path"]).resolve()
        if not candidate.is_relative_to(root):
            raise ManifestError(f"package file escapes bundle: {record['path']}")
        if not candidate.is_file():
            raise ManifestError(f"package file is missing: {record['path']}")
        actual_size = candidate.stat().st_size
        if actual_size != record["size_bytes"]:
            raise ManifestError(
                f"package file size mismatch: {record['path']} "
                f"({actual_size} != {record['size_bytes']})"
            )
        actual_digest = sha256_file(candidate)
        expected_digest = record["digest"]["value"]
        if actual_digest != expected_digest:
            raise ManifestError(f"package file hash mismatch: {record['path']}")


def _validate_capabilities(manifest: dict[str, Any]) -> None:
    if manifest["backend"]["kind"] != "onnx-runtime":
        raise CapabilityError("shipped plans support only ONNX Runtime")
    required = {"device-resident-handoff", "iobinding", "external-data"}
    available = set(manifest["backend"]["capabilities"])
    missing = sorted(required - available)
    if missing:
        raise CapabilityError(f"manifest lacks required capabilities: {missing}")
    if manifest["backend"]["execution_provider"] != "CUDAExecutionProvider":
        raise CapabilityError("shipped plans require CUDAExecutionProvider")


def validate_manifest_package(
    manifest_path: str | Path, *, expected_plan_id: str | None = None
) -> ResolvedPlan:
    """Validate one v2 manifest and the package facts used by the runtime."""

    path = Path(manifest_path).resolve()
    manifest = _load_json(path)
    format_name = manifest.get("format")
    if format_name == MANIFEST_FORMAT_V1:
        raise LegacyManifestError(
            "sam3-split-onnx-v1 is a separate legacy format, not partial v2"
        )
    if format_name != MANIFEST_FORMAT_V2:
        raise ManifestError(f"unsupported manifest format: {format_name!r}")

    schema = _load_json(_schema_path())
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(item) for item in first.absolute_path) or "<root>"
        raise ManifestError(f"manifest schema error at {location}: {first.message}")

    plan_id = str(manifest["plan"]["id"])
    if expected_plan_id is not None and plan_id != expected_plan_id:
        raise ManifestError(
            f"manifest plan ID mismatch: {plan_id!r} != {expected_plan_id!r}"
        )
    if plan_id not in SUPPORTED_PLAN_IDS:
        raise PlanNotFoundError(f"unsupported deployment plan: {plan_id}")

    bundle_dir = path.parent.parent
    _validate_references(manifest, bundle_dir)
    _validate_files(manifest, bundle_dir)
    _validate_capabilities(manifest)
    return ResolvedPlan(
        bundle_dir=bundle_dir,
        manifest_path=path,
        manifest=manifest,
        manifest_digest=sha256_file(path),
    )


def resolve_plan(bundle_dir: str | Path, plan_id: str) -> ResolvedPlan:
    """Resolve a public plan by ID without guessing aliases or legacy metadata."""

    if plan_id not in SUPPORTED_PLAN_IDS:
        raise PlanNotFoundError(f"unknown deployment plan: {plan_id}")
    bundle = Path(bundle_dir).resolve()
    path = bundle / "manifests" / f"{plan_id}.json"
    if not path.is_file():
        raise PlanNotFoundError(f"plan is not present in bundle: {plan_id}")
    return validate_manifest_package(path, expected_plan_id=plan_id)


__all__ = [
    "CapabilityError",
    "BASE_VIDEO_PLAN_ID",
    "BASE_VIDEO_PLAN_IDS",
    "DEFAULT_PLAN_ID",
    "IMAGE_PCS_PLAN_IDS",
    "INTERACTIVE_PLAN_ID",
    "INTERACTIVE_PLAN_IDS",
    "LegacyManifestError",
    "MANIFEST_FORMAT_V1",
    "MANIFEST_FORMAT_V2",
    "ManifestError",
    "PlanNotFoundError",
    "ResolvedPlan",
    "SELECTED_K32_PLAN_ID",
    "SPLIT_PLAN_ID",
    "SUPPORTED_PLAN_IDS",
    "manifest_format",
    "resolve_plan",
    "sha256_file",
    "validate_manifest_package",
]
