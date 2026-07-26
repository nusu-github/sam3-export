"""Export and validate the legacy split ONNX bundle for ``facebook/sam3``.

The bundle implements SAM3 text-only image PCS / legacy split v1. Image
preprocessing, BPE tokenization, score thresholding, selection, optional NMS,
and display-space conversion stay in the Host Runtime. Its four graphs have
fixed batch size one, a 1008x1008 input image, and a 32-token text prompt:

``vision_encoder -> text_encoder -> grounding_encoder -> grounding_decoder``.

Run this on a CUDA machine with the official checkpoint available through the
Hugging Face cache (or pass ``--checkpoint``):

    PYTHONPATH=src python scripts/export_onnx.py --output-dir artifacts/sam3-onnx
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable, Literal, overload

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from sam3.export import GroundingDecode, GroundingEncode, TextTower, VisionTowerFlat
from sam3.grounding.tokenizer_ve import SimpleTokenizer
from sam3.weights.load_sam3 import build_production_text_detector

IMAGE_SIZE = 1008
TEXT_LENGTH = 32
OPSET_VERSION = 18


@dataclass(frozen=True)
class TensorSpec:
    """Portable description of a single static ONNX input or output."""

    name: str
    dtype: str
    shape: list[int]


@dataclass(frozen=True)
class GraphSpec:
    """Portable description of one stage in the split runtime pipeline."""

    file: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]


class GroundingEncodeFlat(nn.Module):
    """Give the one-level encoder an ONNX-friendly flat signature."""

    def __init__(self, cut: GroundingEncode) -> None:
        super().__init__()
        self.cut = cut

    def forward(
        self,
        image_feature_2: Tensor,
        image_pos_2: Tensor,
        image_mask_2: Tensor,
        text_memory: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.cut(
            (image_feature_2,),
            (image_pos_2,),
            (image_mask_2,),
            text_memory,
            text_padding_mask,
        )


class VisionTowerScalpedFlat(nn.Module):
    """Apply the production VL backbone's trailing-FPN ``scalp`` in ONNX."""

    def __init__(self, cut: VisionTowerFlat, scalp: int) -> None:
        super().__init__()
        if scalp < 0 or scalp >= 4:
            raise ValueError(f"scalp must be in [0, 3], got {scalp}")
        self.cut = cut
        self.scalp = int(scalp)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        values = self.cut(pixel_values)
        kept = 4 - self.scalp
        return values[:kept] + values[4 : 4 + kept]


class GroundingDecodeFlat(nn.Module):
    """Give the production three-level decoder an ONNX-friendly flat signature."""

    def __init__(self, cut: GroundingDecode) -> None:
        super().__init__()
        self.cut = cut

    def forward(
        self,
        image_feature_0: Tensor,
        image_feature_1: Tensor,
        image_feature_2: Tensor,
        memory: Tensor,
        pos_embed: Tensor,
        memory_padding_mask: Tensor,
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        encoded_text_memory: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.cut(
            (
                image_feature_0,
                image_feature_1,
                image_feature_2,
            ),
            memory,
            pos_embed,
            memory_padding_mask,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            encoded_text_memory,
            text_padding_mask,
        )


VISION_OUTPUT_NAMES = [
    "image_feature_0",
    "image_feature_1",
    "image_feature_2",
    "image_pos_0",
    "image_pos_1",
    "image_pos_2",
]
ENCODER_OUTPUT_NAMES = [
    "memory",
    "pos_embed",
    "memory_padding_mask",
    "level_start_index",
    "spatial_shapes",
    "valid_ratios",
    "encoded_text_memory",
]


def _tensor_spec(name: str, tensor: Tensor) -> TensorSpec:
    dtype = str(tensor.dtype).removeprefix("torch.")
    return TensorSpec(name=name, dtype=dtype, shape=list(tensor.shape))


def _write_model_card(output_dir: Path, graphs: dict[str, GraphSpec]) -> None:
    card = """---
license: other
tags:
- onnx
- segment-anything
- object-detection
- image-segmentation
---

# SAM3 text-only image PCS / legacy split v1

This repository contains the shipped legacy fixed-shape ONNX bundle derived
from the public [`facebook/sam3`](https://huggingface.co/facebook/sam3)
checkpoint. Its exact scope is **SAM3 text-only image PCS / legacy split v1**.

The legacy bundle consists of four tensor-only stages:

```text
pixels -> vision_encoder.onnx -----> grounding_encoder.onnx -> grounding_decoder.onnx
token ids -> text_encoder.onnx -----^
```

All models use batch size 1, a normalized `1008×1008` RGB image, and a
32-token prompt. `manifest.json` is the source of truth for tensor names,
dtypes, and shapes. The accompanying Space runs these four ONNX files with
ONNX Runtime CUDA Execution Provider and IOBinding on ZeroGPU, so intermediate
vision, text, and grounding tensors remain CUDA `OrtValue`s between graphs.

This package does not provide geometry/exemplar prompts, semantic output,
production interactive PVS, video tracking, or SAM3.1 Tri neck/Multiplex. The
four-stage boundary is a supported legacy recipe and an M1 comparison baseline,
not a declaration of the future default deployment cut. The v1 manifest is the
source of truth only for its recorded filenames and tensor names/dtypes/shapes;
it does not contain v2 plan, cache, capture, fixture or file-integrity metadata.

## Runtime responsibilities

The host must resize/normalize images to `(x / 255 - 0.5) / 0.5`, BPE-tokenize
the text prompt with the SAM 3 tokenizer, pad unused token IDs with zero, and
set `text_padding_mask = input_ids == 0`. It must also apply score thresholds,
NMS if desired, and resize predicted masks to the original image dimensions.
For GPU inference, bind every stage output to CUDA and feed it to the next
stage as an `OrtValue`; copy only final decoder outputs to CPU for rendering.
The accompanying demo thresholds all 200 query outputs, sorts them by score,
and renders at most 25; it does not run NMS.

## License and attribution

These artifacts are derived from Meta's SAM Materials. They are redistributed
under the included [SAM License](LICENSE). Please read and comply with that
license before using or redistributing them. Cite Meta's SAM 3 work and the
source checkpoint in derived work.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    manifest = {
        "format": "sam3-split-onnx-v1",
        "source_model": "facebook/sam3",
        "image_size": IMAGE_SIZE,
        "text_length": TEXT_LENGTH,
        "opset": OPSET_VERSION,
        "graphs": {
            key: {
                **asdict(value),
                "inputs": [asdict(spec) for spec in value.inputs],
                "outputs": [asdict(spec) for spec in value.outputs],
            }
            for key, value in graphs.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _export(
    module: nn.Module,
    args: tuple[Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Export a static graph and run ONNX structural validation."""
    module.eval()
    torch.onnx.export(
        module,
        args,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=True,
    )
    onnx.checker.check_model(path)


def _as_numpy(values: Iterable[Tensor]) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in values]


def _assert_close(
    name: str,
    expected: Iterable[Tensor],
    actual: Iterable[np.ndarray],
    *,
    rtol: float = 5e-2,
    # CUDA EP and PyTorch select different fp16 convolution kernels.  The
    # max observed drift is well below one tenth while masks / scores retain
    # their decisions, so use a tolerance that admits this expected noise.
    atol: float = 1e-1,
) -> None:
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
        np.testing.assert_allclose(
            actual_value,
            expected_value.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} output {index}",
        )


def _session(path: Path) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDAExecutionProvider did not load for {path.name}")
    return session


def _cuda_ortvalue(values: np.ndarray) -> ort.OrtValue:
    """Upload a host tensor once so split graphs can share it on CUDA."""
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(values), "cuda", 0)


@overload
def _run_iobound(
    session: ort.InferenceSession,
    inputs: dict[str, ort.OrtValue],
    *,
    copy_outputs_to_cpu: Literal[False] = False,
) -> list[ort.OrtValue]: ...


@overload
def _run_iobound(
    session: ort.InferenceSession,
    inputs: dict[str, ort.OrtValue],
    *,
    copy_outputs_to_cpu: Literal[True],
) -> list[np.ndarray]: ...


def _run_iobound(
    session: ort.InferenceSession,
    inputs: dict[str, ort.OrtValue],
    *,
    copy_outputs_to_cpu: bool = False,
) -> list[ort.OrtValue] | list[np.ndarray]:
    """Execute a graph with CUDA OrtValue inputs and CUDA-bound outputs."""
    binding = session.io_binding()
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    for output in session.get_outputs():
        binding.bind_output(output.name, "cuda", 0)
    session.run_with_iobinding(binding)
    if copy_outputs_to_cpu:
        return binding.copy_outputs_to_cpu()
    outputs = binding.get_outputs()
    non_cuda = [
        value.device_name() for value in outputs if value.device_name() != "cuda"
    ]
    if non_cuda:
        raise RuntimeError(f"IOBinding output did not remain on CUDA: {non_cuda}")
    return outputs


def _verify(
    output_dir: Path,
    inputs: dict[str, tuple[Tensor, ...]],
    expected: dict[str, tuple[Tensor, ...]],
) -> None:
    """Run all four generated artifacts as a single ONNX Runtime pipeline."""
    vision = _session(output_dir / "vision_encoder.onnx")
    text = _session(output_dir / "text_encoder.onnx")
    encoder = _session(output_dir / "grounding_encoder.onnx")
    decoder = _session(output_dir / "grounding_decoder.onnx")

    pixels = _as_numpy(inputs["vision"])
    vision_out = _run_iobound(vision, {"pixel_values": _cuda_ortvalue(pixels[0])})
    _assert_close(
        "vision", expected["vision"], (output.numpy() for output in vision_out)
    )

    token_ids, attention_mask = _as_numpy(inputs["text"])
    text_out = _run_iobound(
        text,
        {
            "input_ids": _cuda_ortvalue(token_ids),
            "attention_mask": _cuda_ortvalue(attention_mask),
        },
    )
    _assert_close("text", expected["text"], (output.numpy() for output in text_out))

    image_mask = _as_numpy(inputs["encoder"])[2]
    encoder_out = _run_iobound(
        encoder,
        {
            "image_feature_2": vision_out[2],
            "image_pos_2": vision_out[5],
            "image_mask_2": _cuda_ortvalue(image_mask),
            "text_memory": text_out[0],
            "text_padding_mask": text_out[1],
        },
    )
    _assert_close(
        "grounding encoder",
        expected["encoder"],
        (output.numpy() for output in encoder_out),
    )

    decoder_out = _run_iobound(
        decoder,
        {
            **{f"image_feature_{i}": vision_out[i] for i in range(3)},
            "memory": encoder_out[0],
            "pos_embed": encoder_out[1],
            "memory_padding_mask": encoder_out[2],
            "level_start_index": encoder_out[3],
            "spatial_shapes": encoder_out[4],
            "valid_ratios": encoder_out[5],
            "encoded_text_memory": encoder_out[6],
            "text_padding_mask": text_out[1],
        },
        copy_outputs_to_cpu=True,
    )
    # FP16 mask logits accumulate noticeably more CUDA-kernel drift than the
    # detector's scores, boxes, or presence logit. Keep the normal contract
    # tolerance for those decision tensors and a separately bounded tolerance
    # for the dense 288x288 mask field.
    _assert_close(
        "grounding decoder decision tensors",
        (expected["decoder"][0], expected["decoder"][1], expected["decoder"][3]),
        (decoder_out[0], decoder_out[1], decoder_out[3]),
    )
    _assert_close(
        "grounding decoder mask logits",
        (expected["decoder"][2],),
        (decoder_out[2],),
        rtol=1e-1,
        atol=1.0,
    )


def export_onnx(
    output_dir: Path,
    checkpoint: str | None,
    *,
    verify: bool,
) -> None:
    """Build SAM 3, export every graph, and optionally validate the full chain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_production_text_detector(
        add_sam2_neck=False,
        checkpoint_path=checkpoint,
        device="cuda",
        dtype="fp16",
        load_weights=True,
    ).eval()

    vision = VisionTowerScalpedFlat(
        VisionTowerFlat(model.backbone.vision_backbone), model.backbone.scalp
    ).eval()
    text = TextTower(model.backbone.language_backbone).eval()
    encoder = GroundingEncode(model.transformer.encoder, model.num_feature_levels)
    decoder = GroundingDecode(
        model.transformer.decoder,
        model.dot_prod_scoring,
        model.segmentation_head,
    )
    encoder_flat = GroundingEncodeFlat(encoder).eval()
    decoder_flat = GroundingDecodeFlat(decoder).eval()

    tokenizer = SimpleTokenizer()
    token_ids = tokenizer(["a dog"], context_length=TEXT_LENGTH).to("cuda")
    attention_mask = token_ids.ne(0)
    pixels = torch.zeros(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE), device="cuda", dtype=torch.float16
    )

    with torch.inference_mode():
        vision_expected = vision(pixels)
        text_expected = text(token_ids, attention_mask)
        image_mask = torch.zeros((1, 72, 72), device="cuda", dtype=torch.bool)
        encoder_expected = encoder_flat(
            vision_expected[2],
            vision_expected[5],
            image_mask,
            *text_expected,
        )
        decoder_expected = decoder_flat(
            *vision_expected[:3], *encoder_expected, text_expected[1]
        )

    _export(
        vision,
        (pixels,),
        output_dir / "vision_encoder.onnx",
        ["pixel_values"],
        VISION_OUTPUT_NAMES,
    )
    _export(
        text,
        (token_ids, attention_mask),
        output_dir / "text_encoder.onnx",
        ["input_ids", "attention_mask"],
        ["text_memory", "text_padding_mask"],
    )
    _export(
        encoder_flat,
        (
            vision_expected[2],
            vision_expected[5],
            image_mask,
            *text_expected,
        ),
        output_dir / "grounding_encoder.onnx",
        [
            "image_feature_2",
            "image_pos_2",
            "image_mask_2",
            "text_memory",
            "text_padding_mask",
        ],
        ENCODER_OUTPUT_NAMES,
    )
    _export(
        decoder_flat,
        (*vision_expected[:3], *encoder_expected, text_expected[1]),
        output_dir / "grounding_decoder.onnx",
        [
            "image_feature_0",
            "image_feature_1",
            "image_feature_2",
            "memory",
            "pos_embed",
            "memory_padding_mask",
            "level_start_index",
            "spatial_shapes",
            "valid_ratios",
            "encoded_text_memory",
            "text_padding_mask",
        ],
        ["logits", "boxes_cxcywh", "mask_logits", "presence_logits"],
    )

    graphs = {
        "vision": GraphSpec(
            file="vision_encoder.onnx",
            inputs=[_tensor_spec("pixel_values", pixels)],
            outputs=[
                _tensor_spec(name, value)
                for name, value in zip(VISION_OUTPUT_NAMES, vision_expected)
            ],
        ),
        "text": GraphSpec(
            file="text_encoder.onnx",
            inputs=[
                _tensor_spec("input_ids", token_ids),
                _tensor_spec("attention_mask", attention_mask),
            ],
            outputs=[
                _tensor_spec(name, value)
                for name, value in zip(
                    ("text_memory", "text_padding_mask"), text_expected
                )
            ],
        ),
        "grounding_encoder": GraphSpec(
            file="grounding_encoder.onnx",
            inputs=[
                _tensor_spec("image_feature_2", vision_expected[2]),
                _tensor_spec("image_pos_2", vision_expected[5]),
                _tensor_spec("image_mask_2", image_mask),
                _tensor_spec("text_memory", text_expected[0]),
                _tensor_spec("text_padding_mask", text_expected[1]),
            ],
            outputs=[
                _tensor_spec(name, value)
                for name, value in zip(ENCODER_OUTPUT_NAMES, encoder_expected)
            ],
        ),
        "grounding_decoder": GraphSpec(
            file="grounding_decoder.onnx",
            inputs=[
                *[
                    _tensor_spec(f"image_feature_{index}", value)
                    for index, value in enumerate(vision_expected[:3])
                ],
                *[
                    _tensor_spec(name, value)
                    for name, value in zip(ENCODER_OUTPUT_NAMES, encoder_expected)
                ],
                _tensor_spec("text_padding_mask", text_expected[1]),
            ],
            outputs=[
                _tensor_spec(name, value)
                for name, value in zip(
                    ("logits", "boxes_cxcywh", "mask_logits", "presence_logits"),
                    decoder_expected,
                )
            ],
        ),
    }
    _write_model_card(output_dir, graphs)
    shutil.copy2(Path("LICENSE"), output_dir / "LICENSE")
    if verify:
        _verify(
            output_dir,
            {
                "vision": (pixels,),
                "text": (token_ids, attention_mask),
                "encoder": (vision_expected[2], vision_expected[5], image_mask),
            },
            {
                "vision": vision_expected,
                "text": text_expected,
                "encoder": encoder_expected,
                "decoder": decoder_expected,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sam3-onnx"),
        help="Directory to populate with ONNX files and the model card.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to facebook/sam3's sam3.pt checkpoint.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the full CUDA ONNX Runtime pipeline parity check.",
    )
    args = parser.parse_args()
    export_onnx(args.output_dir, args.checkpoint, verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
