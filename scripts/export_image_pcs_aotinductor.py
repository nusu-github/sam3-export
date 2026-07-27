"""Build the M6 AOTInductor CUDA evaluation bundle from canonical M2 captures."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from export_image_pcs_v2 import _copy_file, _file_id, _file_record
import torch

from sam3.runtime.manifest import (
    AOTINDUCTOR_PLAN_ID,
    DEFAULT_PLAN_ID,
    validate_manifest_package,
)

ROLES = ("detector-image-encode", "text-encode", "grounding-full")
SCOPE_LABEL = "SAM3 base text-only image PCS / AOTInductor CUDA evaluation v1"


def _write_card(path: Path) -> None:
    path.write_text(
        """# SAM3 base text-only image PCS / AOTInductor CUDA evaluation v1

This M6 candidate is compiled from the saved canonical `ExportedProgram` files
of the shipped M2 fused semantic plan. It uses the same Public API and semantic
tensor/cache contract as the ORT CUDA default, but it is not the default and
does not imply support for interactive image, video, SAM3.1, CPU, or the M2
selected-K/split plans.
""",
        encoding="utf-8",
    )


def _capture_operator_summary(program: torch.export.ExportedProgram) -> dict[str, Any]:
    call_targets = sorted(
        {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
    )
    non_aten = [
        target
        for target in call_targets
        if not target.startswith(("aten.", "prims.", "operator."))
    ]
    return {
        "call_function_targets": call_targets,
        "non_aten_targets": non_aten,
    }


def _compile_roles(
    source_bundle: Path, staging: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    reports: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for role in ROLES:
        capture_relative = f"capture/{role}.pt2"
        program = torch.export.load(source_bundle / capture_relative)
        package_relative = f"packages/{role}.pt2"
        package_path = staging / package_relative
        package_path.parent.mkdir(parents=True, exist_ok=True)
        compiled = torch._inductor.aoti_compile_and_package(  # type: ignore[attr-defined]
            program,
            package_path=package_path,
        )
        if Path(compiled).resolve() != package_path.resolve():
            raise RuntimeError(f"AOTInductor wrote an unexpected package: {compiled}")
        paths[role] = package_relative
        reports[role] = {
            "status": "compiled",
            "canonical_capture": capture_relative,
            "package": package_relative,
            "package_size_bytes": package_path.stat().st_size,
            "operators": _capture_operator_summary(program),
            "compiler_fallback_nodes": [],
            "compiler_fallback_evidence": (
                "aoti_compile_and_package completed without an unsupported-op error"
            ),
        }
    return reports, paths


def _candidate_manifest(
    source: dict[str, Any],
    staging: Path,
    package_paths: dict[str, str],
) -> dict[str, Any]:
    manifest = deepcopy(source)
    manifest["manifest_id"] = "sam3-image-pcs-aotinductor-m6-evaluation-v2"
    manifest["scope"].update(
        {
            "lifecycle": "candidate",
            "dispatch_role": "optional",
            "scope_label": SCOPE_LABEL,
            "exclusions": [
                "geometry/exemplar prompts",
                "semantic output",
                "interactive image PVS",
                "video tracking",
                "SAM3.1",
                "CPU fallback",
                "M2 selected-K and split plans",
                "default dispatch",
            ],
        }
    )
    manifest["plan"]["id"] = AOTINDUCTOR_PLAN_ID
    manifest["backend"] = {
        "kind": "aotinductor",
        "target": "CUDA device 0",
        "execution_provider": "TorchInductorCUDA",
        "runtime_version": torch.__version__,
        "pytorch_version": torch.__version__,
        "exporter_version": torch.__version__,
        "opset": None,
        "capabilities": ["device-resident-handoff"],
    }
    manifest["capture"]["exporter_version"] = torch.__version__
    for artifact in manifest["artifacts"]:
        role = artifact["role"]
        package_relative = package_paths[role]
        artifact["format"] = "aotinductor"
        artifact["entry_file_ref"] = _file_id(package_relative)
        artifact["external_data_file_refs"] = []
    for handoff in manifest["handoffs"]:
        handoff["mechanism"] = "CUDA torch.Tensor"
        handoff["fallback_plan_id"] = None

    fixture = manifest["fixtures"][0]
    fixture["parity"] = [
        {
            "stage": "official-to-local-eager",
            "status": "pass",
            "report_file_ref": _file_id("reports/m1_measurement_report.json"),
        },
        {
            "stage": "local-eager-to-exported-program",
            "status": "pass",
            "report_file_ref": _file_id("reports/export_report.json"),
        },
        {
            "stage": "exported-program-to-backend",
            "status": "not-run",
            "report_file_ref": None,
        },
        {
            "stage": "end-to-end-behavior",
            "status": "not-run",
            "report_file_ref": None,
        },
    ]
    manifest["files"] = [
        _file_record(staging, path.relative_to(staging).as_posix())
        for path in sorted(staging.rglob("*"))
        if path.is_file() and "manifests/" not in path.as_posix()
    ]
    return manifest


def export_bundle(source_bundle: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    source = validate_manifest_package(
        source_bundle / "manifests" / f"{DEFAULT_PLAN_ID}.json",
        expected_plan_id=DEFAULT_PLAN_ID,
    )
    missing_capture = [
        role for role in ROLES if not (source_bundle / f"capture/{role}.pt2").is_file()
    ]
    if missing_capture:
        raise FileNotFoundError(
            f"source M2 bundle lacks canonical captures: {missing_capture}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        for path in (
            "LICENSE",
            "tokenizer/bpe_simple_vocab_16e6.txt.gz",
            "capture/graph_signatures.json",
            "fixtures/cases.json",
            "fixtures/official_reference.npz",
            "fixtures/official_reference.json",
            "reports/m1_measurement_report.json",
        ):
            _copy_file(source_bundle / path, staging / path)
        for image in sorted((source_bundle / "fixtures/images").glob("*")):
            _copy_file(image, staging / "fixtures/images" / image.name)
        for role in ROLES:
            _copy_file(
                source_bundle / f"capture/{role}.pt2",
                staging / f"capture/{role}.pt2",
            )
        _copy_file(Path("LICENSE"), staging / "LICENSE")
        _write_card(staging / "README.md")

        compile_report, package_paths = _compile_roles(source_bundle, staging)
        export_report = {
            "format": "sam3-image-pcs-m6-aotinductor-export-report-v1",
            "status": "compiled",
            "source_plan_id": DEFAULT_PLAN_ID,
            "plan_id": AOTINDUCTOR_PLAN_ID,
            "torch_version": torch.__version__,
            "canonical_source": "saved M2 ExportedProgram files",
            "roles": compile_report,
        }
        (staging / "reports/export_report.json").write_text(
            json.dumps(export_report, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "reports/m6_aotinductor_validation.json").write_text(
            json.dumps(
                {
                    "format": "sam3-image-pcs-m6-aotinductor-validation-v1",
                    "status": "pending",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "manifests").mkdir()
        manifest = _candidate_manifest(source.manifest, staging, package_paths)
        manifest_path = staging / "manifests" / f"{AOTINDUCTOR_PLAN_ID}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        validate_manifest_package(manifest_path, expected_plan_id=AOTINDUCTOR_PLAN_ID)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_image_pcs_aotinductor.py")),
                "--bundle-dir",
                str(staging),
                "--update-report",
            ],
            check=True,
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-bundle",
        type=Path,
        default=Path("artifacts/sam3-image-pcs-ortcuda-v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sam3-image-pcs-aotinductor-v2"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    export_bundle(arguments.source_bundle.resolve(), arguments.output_dir.resolve())
