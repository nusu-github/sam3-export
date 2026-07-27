"""Build the shipped M2 SAM3 base text-only image PCS ORT CUDA bundle.

The release profile is fixed to B1/1008/L32/Q200/FP16. Every graph is captured
with ``torch.export(strict=False)`` and lowered to ONNX opset 18 with external
data. Generated bundles stay under the ignored ``artifacts/`` tree by default.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from m1_experiments import (
    GroundingDecodeFlat,
    GroundingEncodeFlat,
    GroundingFullFlat,
    GroundingMaskSelectedKFlat,
    GroundingQueryCoreFlat,
    VisionTowerScalpedFlat,
    _export_one,
)
import numpy as np
import onnx
from onnx import TensorProto
from PIL import Image
import torch

from sam3.export import (
    GroundingDecode,
    GroundingEncode,
    GroundingEncodeTextOnly,
    GroundingFull,
    GroundingMaskSelectedK,
    GroundingQueryCore,
    TextOnlyPromptEncode,
    TextTower,
    VisionTowerFlat,
    VisionTowerProfiled,
)
from sam3.grounding.tokenizer_ve import DEFAULT_BPE_PATH, SimpleTokenizer
from sam3.runtime.manifest import (
    DEFAULT_PLAN_ID,
    MANIFEST_FORMAT_V2,
    SELECTED_K32_PLAN_ID,
    SPLIT_PLAN_ID,
    sha256_file,
)
from sam3.weights.load_sam3 import (
    build_production_text_detector,
    resolve_sam3_checkpoint,
)

IMAGE_SIZE = 1008
TEXT_LENGTH = 32
QUERY_COUNT = 200
SELECTED_K = 32
PROFILE_ID = "b1-1008-l32-q200-fp16"
CONTRACT_VERSION = "1.0.0"
SCOPE_LABEL = "SAM3 base text-only image PCS / ORT CUDA v1"
EXCLUSIONS = [
    "geometry/exemplar prompts",
    "semantic output",
    "interactive image PVS",
    "video tracking",
    "SAM3.1",
]


GRAPH_NAMES = {
    "detector-image-encode": "detector_image_encode.onnx",
    "text-encode": "text_encoder.onnx",
    "grounding-full": "grounding_full.onnx",
    "grounding-encode": "grounding_encode.onnx",
    "grounding-decode": "grounding_decode.onnx",
    "grounding-query-core": "grounding_query_core.onnx",
    "grounding-mask-selected-k32": "grounding_mask_selected_k32.onnx",
}

PLAN_RECIPES = {
    DEFAULT_PLAN_ID: {
        "dispatch_role": "default",
        "components": ["DetectorImageEncode", "TextEncoder", "GroundingFull"],
        "roles": ["detector-image-encode", "text-encode", "grounding-full"],
        "output_policy": "raw-200",
        "fallback": None,
    },
    SELECTED_K32_PLAN_ID: {
        "dispatch_role": "optional",
        "components": [
            "DetectorImageEncode",
            "TextEncoder",
            "GroundingEncodeTextOnly",
            "GroundingQueryCore",
            "GroundingMaskSelectedK32",
        ],
        "roles": [
            "detector-image-encode",
            "text-encode",
            "grounding-encode",
            "grounding-query-core",
            "grounding-mask-selected-k32",
        ],
        "output_policy": "selected-k32",
        "fallback": DEFAULT_PLAN_ID,
    },
    SPLIT_PLAN_ID: {
        "dispatch_role": "fallback",
        "components": [
            "DetectorImageEncode",
            "TextEncoder",
            "GroundingEncodeTextOnly",
            "GroundingDecode",
        ],
        "roles": [
            "detector-image-encode",
            "text-encode",
            "grounding-encode",
            "grounding-decode",
        ],
        "output_policy": "raw-200",
        "fallback": None,
    },
}


ROLE_COMPONENTS = {
    "detector-image-encode": ["DetectorImageEncode", "VisionTowerProfiled"],
    "text-encode": ["TextEncoder", "TextTower"],
    "grounding-full": ["GroundingEncodeTextOnly", "GroundingDecode"],
    "grounding-encode": ["GroundingEncodeTextOnly"],
    "grounding-decode": ["GroundingDecode"],
    "grounding-query-core": ["GroundingQueryCore"],
    "grounding-mask-selected-k32": ["GroundingMaskSelectedK32"],
}


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _preprocess_fixture(path: Path) -> torch.Tensor:
    image = (
        Image.open(path)
        .convert("RGB")
        .resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    )
    values = np.asarray(image, dtype=np.float32)
    values = (values / 255.0 - 0.5) / 0.5
    return (
        torch.from_numpy(values)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device="cuda", dtype=torch.float16)
    )


def _onnx_dtype(value: int) -> str:
    names = {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.BFLOAT16: "bfloat16",
        TensorProto.INT64: "int64",
        TensorProto.INT32: "int32",
        TensorProto.BOOL: "bool",
    }
    try:
        return names[value]
    except KeyError as exc:
        raise RuntimeError(f"unsupported release tensor dtype: {value}") from exc


def _value_spec(value: Any) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.dim_param:
            shape.append(str(dimension.dim_param))
        else:
            raise RuntimeError(f"unnamed dynamic dimension for {value.name}")
    return {"dtype": _onnx_dtype(tensor_type.elem_type), "shape": shape}


def _graph_signatures(
    bundle_dir: Path, export_graphs: dict[str, Any]
) -> dict[str, Any]:
    graphs: dict[str, Any] = {}
    for role, name in GRAPH_NAMES.items():
        model = onnx.load(bundle_dir / "graphs" / name, load_external_data=False)
        graphs[role] = {
            "path": f"graphs/{name}",
            "inputs": [
                {"name": value.name, **_value_spec(value)}
                for value in model.graph.input
            ],
            "outputs": [
                {"name": value.name, **_value_spec(value)}
                for value in model.graph.output
            ],
            "exported_program": export_graphs[role]["exported_program"],
        }
    return {
        "format": "sam3-image-pcs-graph-signatures-v1",
        "profile_id": PROFILE_ID,
        "capture": "torch.export(strict=False)",
        "opset": 18,
        "graphs": graphs,
    }


def _file_id(path: str) -> str:
    return "file-" + path.lower().replace("/", "-").replace(".", "-").replace("_", "-")


def _file_role(path: str) -> str:
    if path.endswith(".onnx"):
        return "graph"
    if path.endswith(".onnx.data"):
        return "external-data"
    if path == "LICENSE":
        return "license"
    if path == "README.md":
        return "documentation"
    if path.startswith("tokenizer/"):
        return "tokenizer"
    if path.startswith("fixtures/"):
        return "fixture"
    if path.endswith("graph_signatures.json"):
        return "graph-signature"
    if path.startswith("capture/") and path.endswith(".pt2"):
        return "capture"
    if path.startswith("packages/") and path.endswith(".pt2"):
        return "graph"
    if path.endswith("export_report.json"):
        return "export-report"
    return "parity-report"


def _file_record(bundle_dir: Path, path: str) -> dict[str, Any]:
    absolute = bundle_dir / path
    return {
        "id": _file_id(path),
        "path": path,
        "role": _file_role(path),
        "size_bytes": absolute.stat().st_size,
        "digest": {"algorithm": "sha256", "value": sha256_file(absolute)},
    }


def _tensor_details(
    name: str, spec: dict[str, Any], produced: set[str]
) -> dict[str, Any]:
    host_inputs = {
        "pixel_values",
        "input_ids",
        "attention_mask",
        "image_mask_2",
        "selected_indices",
        "valid_mask",
    }
    final_outputs = {
        "logits",
        "boxes_cxcywh",
        "mask_logits",
        "presence_logits",
        "selected_mask_logits",
    }
    if name in host_inputs and name not in produced:
        residency = "host-input"
    elif name in final_outputs:
        residency = "host-output"
    else:
        residency = "device"

    value_kind = "feature"
    if "mask" in name and "padding" not in name:
        value_kind = "mask-logit" if "logit" in name else "validity-mask"
    elif "logit" in name:
        value_kind = "score-logit"
    elif "box" in name:
        value_kind = "box"
    elif "indices" in name or "index" in name or "shapes" in name:
        value_kind = "index"

    normalization = "none"
    coordinates = "not-applicable"
    unit = "unitless"
    if name == "pixel_values":
        normalization = "(x / 255 - 0.5) / 0.5"
        coordinates = "RGB image resized bilinearly to 1008x1008"
    elif name == "boxes_cxcywh":
        coordinates = "normalized cxcywh"
    elif "mask_logits" in name:
        coordinates = "native 288x288 mask grid"

    return {
        "id": name.replace("_", "-"),
        "semantic_type": name,
        **spec,
        "layout": "static profile; see shape",
        "unit": unit,
        "normalization": normalization,
        "padding": "zero padding where applicable",
        "validity": "fixed profile; validity masks are explicit",
        "coordinates": coordinates,
        "value_kind": value_kind,
        "residency": residency,
    }


def _plan_artifacts(
    recipe: dict[str, Any], signatures: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roles = recipe["roles"]
    produced = {
        output["name"]
        for role in roles
        for output in signatures["graphs"][role]["outputs"]
    }
    specs: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for role in roles:
        signature = signatures["graphs"][role]
        graph_path = signature["path"]
        external_path = graph_path + ".data"
        external_refs = (
            [_file_id(external_path)]
            if (Path(graph_path).parent / (Path(graph_path).name + ".data"))
            else []
        )
        for value in [*signature["inputs"], *signature["outputs"]]:
            spec = {"dtype": value["dtype"], "shape": value["shape"]}
            previous = specs.setdefault(value["name"], spec)
            if previous != spec:
                raise RuntimeError(f"conflicting tensor contract for {value['name']}")
        artifacts.append(
            {
                "id": role,
                "role": role,
                "format": "onnx",
                "components": ROLE_COMPONENTS[role],
                "entry_file_ref": _file_id(graph_path),
                "external_data_file_refs": external_refs,
                "inputs": [
                    {
                        "tensor_ref": value["name"].replace("_", "-"),
                        "backend_name": value["name"],
                    }
                    for value in signature["inputs"]
                ],
                "outputs": [
                    {
                        "tensor_ref": value["name"].replace("_", "-"),
                        "backend_name": value["name"],
                    }
                    for value in signature["outputs"]
                ],
            }
        )
    tensors = [
        _tensor_details(name, spec, produced) for name, spec in sorted(specs.items())
    ]
    return artifacts, tensors


def _edges_for(plan_id: str) -> list[dict[str, Any]]:
    def edge(producer: str, consumer: str, names: Iterable[str]) -> dict[str, Any]:
        return {
            "producer_artifact_ref": producer,
            "consumer_artifact_ref": consumer,
            "tensor_refs": [name.replace("_", "-") for name in names],
        }

    common = [
        edge(
            "detector-image-encode",
            "grounding-encode" if plan_id != DEFAULT_PLAN_ID else "grounding-full",
            ["image_feature_2", "image_pos_2"],
        ),
        edge(
            "text-encode",
            "grounding-encode" if plan_id != DEFAULT_PLAN_ID else "grounding-full",
            ["text_memory", "text_padding_mask"],
        ),
    ]
    if plan_id == DEFAULT_PLAN_ID:
        common.append(
            edge(
                "detector-image-encode",
                "grounding-full",
                ["image_feature_0", "image_feature_1"],
            )
        )
        return common
    if plan_id == SPLIT_PLAN_ID:
        return [
            *common,
            edge(
                "detector-image-encode",
                "grounding-decode",
                ["image_feature_0", "image_feature_1", "image_feature_2"],
            ),
            edge(
                "grounding-encode",
                "grounding-decode",
                [
                    "memory",
                    "pos_embed",
                    "memory_padding_mask",
                    "level_start_index",
                    "spatial_shapes",
                    "valid_ratios",
                    "prompt_memory",
                    "prompt_padding_mask",
                ],
            ),
        ]
    return [
        *common,
        edge(
            "grounding-encode",
            "grounding-query-core",
            [
                "memory",
                "pos_embed",
                "memory_padding_mask",
                "level_start_index",
                "spatial_shapes",
                "valid_ratios",
                "prompt_memory",
                "prompt_padding_mask",
            ],
        ),
        edge(
            "detector-image-encode",
            "grounding-mask-selected-k32",
            ["image_feature_0", "image_feature_1", "image_feature_2"],
        ),
        edge(
            "grounding-encode",
            "grounding-mask-selected-k32",
            ["memory", "prompt_memory", "prompt_padding_mask"],
        ),
        edge(
            "grounding-query-core",
            "grounding-mask-selected-k32",
            ["query_embeddings"],
        ),
    ]


def _manifest(
    bundle_dir: Path,
    plan_id: str,
    signatures: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    recipe = PLAN_RECIPES[plan_id]
    artifacts, tensors = _plan_artifacts(recipe, signatures)
    file_paths = {
        "LICENSE",
        "README.md",
        "tokenizer/bpe_simple_vocab_16e6.txt.gz",
        "capture/graph_signatures.json",
        "reports/export_report.json",
        "reports/m1_export_report.json",
        "reports/m1_measurement_report.json",
        "reports/m2_release_validation.json",
        "fixtures/cases.json",
        "fixtures/official_reference.npz",
        "fixtures/official_reference.json",
    }
    for image in metadata["fixtures"]["images"]:
        file_paths.add(f"fixtures/images/{Path(image['workspace_path']).name}")
    for role in recipe["roles"]:
        file_paths.add(signatures["graphs"][role]["exported_program"]["program_path"])
        graph_path = signatures["graphs"][role]["path"]
        file_paths.add(graph_path)
        external_path = graph_path + ".data"
        if (bundle_dir / external_path).is_file():
            file_paths.add(external_path)
    files = [_file_record(bundle_dir, path) for path in sorted(file_paths)]

    image_tensor_refs = [
        value["tensor_ref"]
        for artifact in artifacts
        if artifact["role"] == "detector-image-encode"
        for value in artifact["outputs"]
    ]
    text_tensor_refs = [
        value["tensor_ref"]
        for artifact in artifacts
        if artifact["role"] == "text-encode"
        for value in artifact["outputs"]
    ]
    handoffs = [
        {
            "id": f"handoff-{index}",
            "producer_artifact_ref": edge["producer_artifact_ref"],
            "consumer_artifact_ref": edge["consumer_artifact_ref"],
            "tensor_refs": edge["tensor_refs"],
            "requirement": "required-device",
            "mechanism": "ORT CUDA IOBinding / CUDA OrtValue",
            "fallback_plan_id": recipe["fallback"],
        }
        for index, edge in enumerate(_edges_for(plan_id), start=1)
    ]
    parity_ref = _file_id("reports/m1_measurement_report.json")
    release_ref = _file_id("reports/m2_release_validation.json")
    return {
        "format": MANIFEST_FORMAT_V2,
        "manifest_id": f"{plan_id}-manifest-v2",
        "scope": {
            "classification": "public-deployment",
            "lifecycle": "shipped",
            "dispatch_role": recipe["dispatch_role"],
            "scope_label": SCOPE_LABEL,
            "use_case": "image-pcs",
            "prompt_coverage": ["text"],
            "capabilities": ["image PCS", "raw query output", recipe["output_policy"]],
            "exclusions": EXCLUSIONS,
        },
        "plan": {
            "id": plan_id,
            "contract_version": CONTRACT_VERSION,
            "semantic_graph_kind": "sam3-base-text-only-image-pcs",
            "role_set": recipe["roles"],
            "components": recipe["components"],
        },
        "model": {
            "family": "sam3",
            "variant": "base",
            "vision_layout": "three detector FPN levels; required final position",
            "tracking_layout": "not-applicable",
            "source_repository": str(metadata["official_repository"]),
            "source_commit": metadata["official_commit"],
            "model_revision": metadata["model_revision"],
            "checkpoint": {
                "id": "facebook-sam3-sam3-pt",
                "digest": {
                    "algorithm": "sha256",
                    "value": metadata["checkpoint_sha256"],
                },
            },
            "variant_parameters": [],
        },
        "backend": {
            "kind": "onnx-runtime",
            "target": "CUDA device 0",
            "execution_provider": "CUDAExecutionProvider",
            "runtime_version": "1.27.0",
            "pytorch_version": metadata["torch_version"],
            "exporter_version": metadata["onnx_version"],
            "opset": 18,
            "capabilities": [
                "device-resident-handoff",
                "iobinding",
                "external-data",
            ],
        },
        "profile": {
            "id": PROFILE_ID,
            "precision": "fp16",
            "shape_mode": "static",
            "static_values": [
                {"name": "batch", "value": 1},
                {"name": "image-size", "value": IMAGE_SIZE},
                {"name": "text-length", "value": TEXT_LENGTH},
                {"name": "queries", "value": QUERY_COUNT},
                {
                    "name": "selected-k",
                    "value": SELECTED_K if plan_id == SELECTED_K32_PLAN_ID else 0,
                },
            ],
            "dynamic_dimensions": [],
        },
        "tensors": tensors,
        "artifacts": artifacts,
        "execution": {
            "entry_artifacts": ["detector-image-encode", "text-encode"],
            "edges": _edges_for(plan_id),
        },
        "caches": [
            {
                "id": "image-cache",
                "tensor_refs": image_tensor_refs,
                "lifetime": "image-session",
                "key_version": "1.0.0",
                "key_parts": [
                    "preprocessed-bytes",
                    "original-size",
                    "checkpoint-digest",
                    "profile-id",
                ],
                "invalidated_by": [
                    "image-change",
                    "checkpoint-change",
                    "profile-change",
                ],
                "state_compatibility": "matching checkpoint and static profile",
            },
            {
                "id": "prompt-cache",
                "tensor_refs": text_tensor_refs,
                "lifetime": "prompt-session",
                "key_version": "1.0.0",
                "key_parts": [
                    "token-ids",
                    "tokenizer-digest",
                    "checkpoint-digest",
                    "model-revision",
                    "profile-id",
                ],
                "invalidated_by": ["text-change", "tokenizer-change", "model-change"],
                "state_compatibility": "matching tokenizer, checkpoint, and profile",
            },
        ],
        "handoffs": handoffs,
        "capture": {
            "canonical_format": "exported-program",
            "mode": "non-strict",
            "pytorch_version": metadata["torch_version"],
            "exporter_version": metadata["onnx_version"],
            "constraints": [
                "batch=1",
                "image=1008x1008",
                "text-length=32",
                "queries=200",
            ],
            "graph_signature_file_ref": _file_id("capture/graph_signatures.json"),
            "program_file_refs": [
                _file_id(signatures["graphs"][role]["exported_program"]["program_path"])
                for role in recipe["roles"]
            ],
            "strict_audit": {"status": "not-run", "report_file_ref": None},
        },
        "policies": [
            {
                "name": "output-policy",
                "binding": "baked",
                "value": recipe["output_policy"],
                "stage": "artifact-plan",
            },
            {
                "name": "score-policy",
                "binding": "host-runtime",
                "value": "sigmoid(logit) * sigmoid(presence); strict > threshold",
                "stage": "public-result",
            },
            {
                "name": "selection-policy",
                "binding": "host-runtime",
                "value": "score descending, query index ascending; optional mask NMS",
                "stage": "public-result",
            },
        ],
        "fixtures": [
            {
                "id": "m1-image-pcs-v1",
                "version": "1.0.0",
                "owner": {"team": "sam3-export", "role": "Export Tech Lead"},
                "source_commit": metadata["official_commit"],
                "checkpoint_digest": {
                    "algorithm": "sha256",
                    "value": metadata["checkpoint_sha256"],
                },
                "cases": [case["id"] for case in metadata["fixtures"]["cases"]],
                "aggregate_digest": {
                    "algorithm": "sha256",
                    "value": sha256_file(bundle_dir / "fixtures" / "cases.json"),
                },
                "parity": [
                    {
                        "stage": "official-to-local-eager",
                        "status": "pass",
                        "report_file_ref": parity_ref,
                    },
                    {
                        "stage": "local-eager-to-exported-program",
                        "status": "pass",
                        "report_file_ref": _file_id("reports/export_report.json"),
                    },
                    {
                        "stage": "exported-program-to-backend",
                        "status": "pass",
                        "report_file_ref": parity_ref,
                    },
                    {
                        "stage": "end-to-end-behavior",
                        "status": "pass",
                        "report_file_ref": release_ref,
                    },
                ],
            }
        ],
        "files": files,
    }


def refresh_manifest_file_records(bundle_dir: Path) -> None:
    """Refresh file facts after the release validator rewrites its report."""

    for path in sorted((bundle_dir / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"] = [
            _file_record(bundle_dir, record["path"]) for record in manifest["files"]
        ]
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_bundle_card(path: Path) -> None:
    path.write_text(
        """# SAM3 base text-only image PCS / ORT CUDA v1

This M2 bundle ships three manifest-driven plans for the fixed
`b1-1008-l32-q200-fp16` profile: fused raw-200 default, optional selected-K32,
and corrected split fallback. See each file under `manifests/` for the
machine-readable artifact, cache, handoff, policy, fixture, and integrity
contract. The Public API is documented in the source repository's
`docs/IMAGE_PCS_API.md`.

This scope excludes geometry/exemplar prompts, semantic output, interactive
image PVS, video, and SAM3.1. It is separate from the shipped
`SAM3 text-only image PCS / legacy split v1` bundle.
""",
        encoding="utf-8",
    )


def export_bundle(
    output_dir: Path,
    checkpoint_path: Path,
    official_repo: Path,
    fixtures_path: Path,
    m1_work_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"output already exists; choose a new directory or remove it explicitly: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        graphs_dir = staging / "graphs"
        graphs_dir.mkdir()
        fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if checkpoint_sha256 != fixtures["checkpoint_sha256"]:
            raise RuntimeError("checkpoint does not match the owned M1 fixture")
        official_commit = _git_revision(official_repo)
        if official_commit != fixtures["official_source_commit"]:
            raise RuntimeError(
                "official repository commit does not match the M1 fixture"
            )

        workspace = fixtures_path.resolve().parents[4]
        first_case = fixtures["cases"][0]
        image_record = next(
            item for item in fixtures["images"] if item["id"] == first_case["image"]
        )
        fixture_image = workspace / image_record["workspace_path"]
        pixels = _preprocess_fixture(fixture_image)

        model = build_production_text_detector(
            add_sam2_neck=False,
            checkpoint_path=str(checkpoint_path),
            device="cuda",
            dtype="fp16",
            load_weights=True,
        ).eval()
        vision = VisionTowerProfiled(
            VisionTowerScalpedFlat(
                VisionTowerFlat(model.backbone.vision_backbone), model.backbone.scalp
            ),
            "required-position-only",
        ).eval()
        text = TextTower(model.backbone.language_backbone).eval()
        encoder = GroundingEncodeTextOnly(
            GroundingEncode(model.transformer.encoder, model.num_feature_levels),
            TextOnlyPromptEncode(model.geometry_encoder),
        ).eval()
        decoder = GroundingDecode(
            model.transformer.decoder,
            model.dot_prod_scoring,
            model.segmentation_head,
        ).eval()
        encoder_flat = GroundingEncodeFlat(encoder).eval()
        decoder_flat = GroundingDecodeFlat(decoder).eval()
        full_flat = GroundingFullFlat(GroundingFull(encoder, decoder)).eval()
        query_flat = GroundingQueryCoreFlat(
            GroundingQueryCore(model.transformer.decoder, model.dot_prod_scoring)
        ).eval()
        mask_flat = GroundingMaskSelectedKFlat(
            GroundingMaskSelectedK(model.segmentation_head)
        ).eval()

        tokenizer = SimpleTokenizer()
        token_ids = tokenizer([first_case["text"]], context_length=TEXT_LENGTH).to(
            "cuda"
        )
        attention_mask = token_ids.ne(0)
        image_mask = torch.zeros((1, 72, 72), device="cuda", dtype=torch.bool)
        with torch.inference_mode():
            vision_outputs = vision(pixels)
            text_outputs = text(token_ids, attention_mask)
            encoded = encoder_flat(
                vision_outputs[2], vision_outputs[3], image_mask, *text_outputs
            )
            query_args = (vision_outputs[2], *encoded[:7], encoded[7])
            query_outputs = query_flat(*query_args)

        encoder_output_names = [
            "memory",
            "pos_embed",
            "memory_padding_mask",
            "level_start_index",
            "spatial_shapes",
            "valid_ratios",
            "prompt_memory",
            "prompt_padding_mask",
        ]
        output_names = ["logits", "boxes_cxcywh", "mask_logits", "presence_logits"]
        export_report: dict[str, Any] = {
            "format": "sam3-image-pcs-m2-export-report-v1",
            "profile_id": PROFILE_ID,
            "capture": "torch.export(strict=False)",
            "opset": 18,
            "sam3_export_commit": _git_revision(Path(__file__).resolve().parents[1]),
            "checkpoint_sha256": checkpoint_sha256,
            "official_commit": official_commit,
            "graphs": {},
        }
        export_report["graphs"]["detector-image-encode"] = _export_one(
            vision,
            (pixels,),
            graphs_dir / GRAPH_NAMES["detector-image-encode"],
            ["pixel_values"],
            ["image_feature_0", "image_feature_1", "image_feature_2", "image_pos_2"],
            capture_path=staging / "capture/detector-image-encode.pt2",
            capture_bundle_path="capture/detector-image-encode.pt2",
        )
        export_report["graphs"]["text-encode"] = _export_one(
            text,
            (token_ids, attention_mask),
            graphs_dir / GRAPH_NAMES["text-encode"],
            ["input_ids", "attention_mask"],
            ["text_memory", "text_padding_mask"],
            capture_path=staging / "capture/text-encode.pt2",
            capture_bundle_path="capture/text-encode.pt2",
        )
        common_args = (
            *vision_outputs[:3],
            vision_outputs[3],
            image_mask,
            *text_outputs,
        )
        export_report["graphs"]["grounding-full"] = _export_one(
            full_flat,
            common_args,
            graphs_dir / GRAPH_NAMES["grounding-full"],
            [
                "image_feature_0",
                "image_feature_1",
                "image_feature_2",
                "image_pos_2",
                "image_mask_2",
                "text_memory",
                "text_padding_mask",
            ],
            output_names,
            capture_path=staging / "capture/grounding-full.pt2",
            capture_bundle_path="capture/grounding-full.pt2",
        )
        export_report["graphs"]["grounding-encode"] = _export_one(
            encoder_flat,
            (vision_outputs[2], vision_outputs[3], image_mask, *text_outputs),
            graphs_dir / GRAPH_NAMES["grounding-encode"],
            [
                "image_feature_2",
                "image_pos_2",
                "image_mask_2",
                "text_memory",
                "text_padding_mask",
            ],
            encoder_output_names,
            capture_path=staging / "capture/grounding-encode.pt2",
            capture_bundle_path="capture/grounding-encode.pt2",
        )
        export_report["graphs"]["grounding-decode"] = _export_one(
            decoder_flat,
            (*vision_outputs[:3], *encoded[:7], encoded[7]),
            graphs_dir / GRAPH_NAMES["grounding-decode"],
            [
                "image_feature_0",
                "image_feature_1",
                "image_feature_2",
                *encoder_output_names,
            ],
            output_names,
            capture_path=staging / "capture/grounding-decode.pt2",
            capture_bundle_path="capture/grounding-decode.pt2",
        )
        export_report["graphs"]["grounding-query-core"] = _export_one(
            query_flat,
            query_args,
            graphs_dir / GRAPH_NAMES["grounding-query-core"],
            [
                "image_feature_2",
                "memory",
                "pos_embed",
                "memory_padding_mask",
                "level_start_index",
                "spatial_shapes",
                "valid_ratios",
                "prompt_memory",
                "prompt_padding_mask",
            ],
            ["logits", "boxes_cxcywh", "presence_logits", "query_embeddings"],
            capture_path=staging / "capture/grounding-query-core.pt2",
            capture_bundle_path="capture/grounding-query-core.pt2",
        )
        indices = torch.arange(SELECTED_K, device="cuda", dtype=torch.int64).unsqueeze(
            0
        )
        valid = torch.ones((1, SELECTED_K), device="cuda", dtype=torch.bool)
        export_report["graphs"]["grounding-mask-selected-k32"] = _export_one(
            mask_flat,
            (
                *vision_outputs[:3],
                encoded[0],
                encoded[6],
                encoded[7],
                query_outputs[3],
                indices,
                valid,
            ),
            graphs_dir / GRAPH_NAMES["grounding-mask-selected-k32"],
            [
                "image_feature_0",
                "image_feature_1",
                "image_feature_2",
                "memory",
                "prompt_memory",
                "prompt_padding_mask",
                "query_embeddings",
                "selected_indices",
                "valid_mask",
            ],
            ["selected_mask_logits"],
            capture_path=staging / "capture/grounding-mask-selected-k32.pt2",
            capture_bundle_path="capture/grounding-mask-selected-k32.pt2",
        )

        _copy_file(Path("LICENSE"), staging / "LICENSE")
        _copy_file(DEFAULT_BPE_PATH, staging / "tokenizer" / DEFAULT_BPE_PATH.name)
        _copy_file(fixtures_path, staging / "fixtures" / "cases.json")
        for image in fixtures["images"]:
            source = workspace / image["workspace_path"]
            if sha256_file(source) != image["sha256"]:
                raise RuntimeError(f"fixture image hash mismatch: {source}")
            _copy_file(source, staging / "fixtures" / "images" / source.name)
        for name in (
            "official_reference.npz",
            "official_reference.json",
            "export_report.json",
            "measurement_report.json",
        ):
            if not (m1_work_dir / name).is_file():
                raise FileNotFoundError(
                    f"required replayable M1 result is missing: {m1_work_dir / name}"
                )
        _copy_file(
            m1_work_dir / "official_reference.npz",
            staging / "fixtures" / "official_reference.npz",
        )
        _copy_file(
            m1_work_dir / "official_reference.json",
            staging / "fixtures" / "official_reference.json",
        )
        _copy_file(
            m1_work_dir / "export_report.json",
            staging / "reports" / "m1_export_report.json",
        )
        _copy_file(
            m1_work_dir / "measurement_report.json",
            staging / "reports" / "m1_measurement_report.json",
        )
        (staging / "reports" / "export_report.json").write_text(
            json.dumps(export_report, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "reports" / "m2_release_validation.json").write_text(
            json.dumps(
                {
                    "format": "sam3-image-pcs-m2-release-validation-v1",
                    "status": "pending",
                    "reason": "rewritten by scripts/validate_image_pcs_v2.py",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        signatures = _graph_signatures(staging, export_report["graphs"])
        (staging / "capture" / "graph_signatures.json").write_text(
            json.dumps(signatures, indent=2) + "\n", encoding="utf-8"
        )
        _write_bundle_card(staging / "README.md")
        metadata = {
            "official_repository": "facebook/sam3",
            "official_commit": official_commit,
            "model_revision": official_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "fixtures": fixtures,
        }
        (staging / "manifests").mkdir()
        for plan_id in PLAN_RECIPES:
            manifest = _manifest(staging, plan_id, signatures, metadata)
            (staging / "manifests" / f"{plan_id}.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_image_pcs_v2.py")),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sam3-image-pcs-ortcuda-v2"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--official-repo", type=Path, default=Path("../sam3"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("tests/fixtures/m1_image_pcs/cases.json"),
    )
    parser.add_argument("--m1-work-dir", type=Path, default=Path(".m1-work"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = resolve_sam3_checkpoint(
        str(args.checkpoint) if args.checkpoint is not None else None
    )
    export_bundle(
        args.output_dir.resolve(),
        checkpoint,
        args.official_repo.resolve(),
        args.fixtures.resolve(),
        args.m1_work_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
