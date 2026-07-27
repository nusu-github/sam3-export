"""Build the M6 direct ExportedProgram CUDA evaluation bundle."""

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
    DEFAULT_PLAN_ID,
    EXPORTED_PROGRAM_PLAN_ID,
    validate_manifest_package,
)

ROLES = ("detector-image-encode", "text-encode", "grounding-full")
SCOPE_LABEL = "SAM3 base text-only image PCS / ExportedProgram CUDA evaluation v1"


def _write_card(path: Path) -> None:
    path.write_text(
        """# SAM3 base text-only image PCS / ExportedProgram CUDA evaluation v1

This M6 candidate executes the saved canonical `ExportedProgram` files with
PyTorch ATen CUDA. It preserves the M2 fused plan's Public API and semantic
tensor/cache contract. It is optional, does not replace the ORT CUDA default,
and does not imply support for other capabilities or profiles.
""",
        encoding="utf-8",
    )


def _manifest(source: dict[str, Any], staging: Path) -> dict[str, Any]:
    manifest = deepcopy(source)
    manifest["manifest_id"] = "sam3-image-pcs-exported-program-m6-evaluation-v2"
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
    manifest["plan"]["id"] = EXPORTED_PROGRAM_PLAN_ID
    manifest["backend"] = {
        "kind": "exported-program",
        "target": "CUDA device 0",
        "execution_provider": "PyTorchATenCUDA",
        "runtime_version": torch.__version__,
        "pytorch_version": torch.__version__,
        "exporter_version": torch.__version__,
        "opset": None,
        "capabilities": ["device-resident-handoff"],
    }
    manifest["capture"]["exporter_version"] = torch.__version__
    for artifact in manifest["artifacts"]:
        role = artifact["role"]
        capture_path = f"capture/{role}.pt2"
        artifact["format"] = "exported-program"
        artifact["entry_file_ref"] = _file_id(capture_path)
        artifact["external_data_file_refs"] = []
    for handoff in manifest["handoffs"]:
        handoff["mechanism"] = "CUDA torch.Tensor"
        handoff["fallback_plan_id"] = None
    manifest["fixtures"][0]["parity"] = [
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
            source_capture = source_bundle / f"capture/{role}.pt2"
            if not source_capture.is_file():
                raise FileNotFoundError(f"canonical capture is missing: {role}")
            _copy_file(source_capture, staging / f"capture/{role}.pt2")
        _write_card(staging / "README.md")
        export_report = {
            "format": "sam3-image-pcs-m6-exported-program-export-report-v1",
            "status": "captured",
            "source_plan_id": DEFAULT_PLAN_ID,
            "plan_id": EXPORTED_PROGRAM_PLAN_ID,
            "torch_version": torch.__version__,
            "roles": {
                role: {
                    "status": "saved",
                    "path": f"capture/{role}.pt2",
                    "size_bytes": (staging / f"capture/{role}.pt2").stat().st_size,
                }
                for role in ROLES
            },
        }
        (staging / "reports/export_report.json").write_text(
            json.dumps(export_report, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "reports/m6_exported_program_validation.json").write_text(
            json.dumps(
                {
                    "format": "sam3-image-pcs-m6-exported-program-validation-v1",
                    "status": "pending",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "manifests").mkdir()
        manifest_path = staging / "manifests" / f"{EXPORTED_PROGRAM_PLAN_ID}.json"
        manifest_path.write_text(
            json.dumps(_manifest(source.manifest, staging), indent=2) + "\n",
            encoding="utf-8",
        )
        validate_manifest_package(
            manifest_path, expected_plan_id=EXPORTED_PROGRAM_PLAN_ID
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_image_pcs_aotinductor.py")),
                "--bundle-dir",
                str(staging),
                "--plan-id",
                EXPORTED_PROGRAM_PLAN_ID,
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
        default=Path("artifacts/sam3-image-pcs-exported-program-v2"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    export_bundle(arguments.source_bundle.resolve(), arguments.output_dir.resolve())
