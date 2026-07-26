"""ZeroGPU demo for SAM3 text-only image PCS / legacy split v1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, overload

# ZeroGPU must patch CUDA before any dependency can touch it.
import spaces  # isort: skip

import gradio as gr
from huggingface_hub import snapshot_download
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from tokenizers import Tokenizer

MODEL_ID = os.environ.get("SAM3_ONNX_MODEL_ID", "RamRom/sam3-onnx")
IMAGE_SIZE = 1008
TEXT_LENGTH = 32


def _model_dir() -> Path:
    """Use a local artifact directory for development or download the Hub repo."""
    local_dir = os.environ.get("SAM3_ONNX_MODEL_DIR")
    if local_dir:
        return Path(local_dir)
    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[
                "*.onnx",
                "*.onnx.data",
                "manifest.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        )
    )


_MODEL_DIR = _model_dir()
_TOKENIZER = Tokenizer.from_file(str(_MODEL_DIR / "tokenizer.json"))
_SESSIONS: dict[str, ort.InferenceSession] | None = None


def _cuda_ortvalue(values: np.ndarray) -> ort.OrtValue:
    """Upload a contiguous NumPy tensor once, as an ORT-owned CUDA value."""
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
    """Run one graph while retaining CUDA OrtValues at every model boundary.

    ``session.run`` materializes NumPy arrays, which would copy each split
    model's output back to host memory.  IOBinding instead lets ORT allocate
    every output on CUDA and feed those same OrtValues to the next graph.
    """
    binding = session.io_binding()
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    for output in session.get_outputs():
        binding.bind_output(output.name, "cuda", 0)
    session.run_with_iobinding(binding)

    if copy_outputs_to_cpu:
        # This is the sole device-to-host transfer: PIL/NumPy post-processing
        # needs the decoder outputs on CPU.
        return binding.copy_outputs_to_cpu()

    outputs = binding.get_outputs()
    non_cuda = [
        value.device_name() for value in outputs if value.device_name() != "cuda"
    ]
    if non_cuda:
        raise RuntimeError(
            f"ONNX Runtime failed to retain a split-model output on CUDA: {non_cuda}"
        )
    return outputs


def _sessions() -> dict[str, ort.InferenceSession]:
    """Create and cache CUDA sessions in the active ZeroGPU worker."""
    global _SESSIONS
    if _SESSIONS is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _SESSIONS = {
            name: ort.InferenceSession(str(_MODEL_DIR / filename), providers=providers)
            for name, filename in {
                "vision": "vision_encoder.onnx",
                "text": "text_encoder.onnx",
                "encoder": "grounding_encoder.onnx",
                "decoder": "grounding_decoder.onnx",
            }.items()
        }
        inactive = [
            name
            for name, session in _SESSIONS.items()
            if "CUDAExecutionProvider" not in session.get_providers()
        ]
        if inactive:
            raise RuntimeError(f"CUDA Execution Provider was unavailable: {inactive}")
    return _SESSIONS


def _preprocess(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None].astype(np.float16))


def _tokenize(prompt: str) -> tuple[np.ndarray, np.ndarray]:
    """Match SAM 3's CLIP BPE IDs and its zero-padding convention exactly."""
    token_ids = _TOKENIZER.encode(prompt).ids[:TEXT_LENGTH]
    token_ids[-1] = 49407
    ids = np.zeros((1, TEXT_LENGTH), dtype=np.int64)
    ids[0, : len(token_ids)] = token_ids
    return ids, ids != 0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return 1.0 / (1.0 + np.exp(-values))


def _overlay(
    image: Image.Image,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
) -> Image.Image:
    canvas = image.convert("RGBA")
    width, height = canvas.size
    palette = [(46, 196, 182), (255, 159, 67), (84, 160, 255), (255, 107, 107)]
    for index, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
        color = palette[index % len(palette)]
        mask_image = Image.fromarray((mask > 0.5).astype(np.uint8) * 112).resize(
            (width, height), Image.Resampling.BILINEAR
        )
        layer = Image.new("RGBA", (width, height), (*color, 0))
        layer.putalpha(mask_image)
        canvas.alpha_composite(layer)
        cx, cy, box_w, box_h = box
        left = (cx - box_w / 2) * width
        top = (cy - box_h / 2) * height
        right = (cx + box_w / 2) * width
        bottom = (cy + box_h / 2) * height
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((left, top, right, bottom), outline=(*color, 255), width=3)
        draw.text((left + 4, max(0, top - 18)), f"{score:.2f}", fill=(*color, 255))
    return canvas.convert("RGB")


@spaces.GPU(duration=90)
def segment(
    image: Image.Image, concept: str, threshold: float
) -> tuple[Image.Image, str]:
    """Segment text matches with the legacy split SAM3 image PCS bundle."""
    if image is None:
        raise gr.Error("Upload an image first.")
    if not concept or not concept.strip():
        raise gr.Error("Enter a text concept, such as 'dog' or 'red car'.")

    sessions = _sessions()
    pixels = _preprocess(image)
    token_ids, attention_mask = _tokenize(concept.strip())
    vision = _run_iobound(sessions["vision"], {"pixel_values": _cuda_ortvalue(pixels)})
    text = _run_iobound(
        sessions["text"],
        {
            "input_ids": _cuda_ortvalue(token_ids),
            "attention_mask": _cuda_ortvalue(attention_mask),
        },
    )
    encoded = _run_iobound(
        sessions["encoder"],
        {
            "image_feature_2": vision[2],
            "image_pos_2": vision[5],
            "image_mask_2": _cuda_ortvalue(np.zeros((1, 72, 72), dtype=np.bool_)),
            "text_memory": text[0],
            "text_padding_mask": text[1],
        },
    )
    decoder_out = _run_iobound(
        sessions["decoder"],
        {
            **{f"image_feature_{index}": vision[index] for index in range(3)},
            "memory": encoded[0],
            "pos_embed": encoded[1],
            "memory_padding_mask": encoded[2],
            "level_start_index": encoded[3],
            "spatial_shapes": encoded[4],
            "valid_ratios": encoded[5],
            "encoded_text_memory": encoded[6],
            "text_padding_mask": text[1],
        },
        copy_outputs_to_cpu=True,
    )
    logits, boxes, mask_logits, presence_logits = decoder_out

    scores = _sigmoid(logits[0, :, 0]) * _sigmoid(presence_logits[0, 0])
    selected = np.flatnonzero(scores >= threshold)
    selected = selected[np.argsort(scores[selected])[::-1]][:25]
    if len(selected) == 0:
        return image.convert("RGB"), "No masks met the selected threshold."
    masks = _sigmoid(mask_logits[0, selected])
    result = _overlay(image, boxes[0, selected], masks, scores[selected])
    summary = "\n".join(
        [
            f"**{len(selected)} match(es)** for `{concept.strip()}`",
            "Scores: " + ", ".join(f"{score:.3f}" for score in scores[selected]),
        ]
    )
    return result, summary


with gr.Blocks(theme=gr.themes.Soft(), title="SAM3 Legacy Split ONNX") as demo:
    gr.Markdown(
        "# SAM3 text-only image PCS / legacy split v1\n"
        "Four CUDA ONNX Runtime graphs on ZeroGPU, chained with device-resident "
        "IOBinding. This demo renders at most 25 of 200 queries and does not run NMS."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil", label="Image", sources=["upload", "clipboard"]
            )
            concept_input = gr.Textbox(
                label="Concept", placeholder="e.g. dog, person, red car"
            )
            threshold_input = gr.Slider(
                0.05, 0.95, value=0.5, step=0.05, label="Score threshold"
            )
            run_button = gr.Button("Segment", variant="primary")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Segmentation")
            summary_output = gr.Markdown()
    run_button.click(
        segment,
        inputs=[image_input, concept_input, threshold_input],
        outputs=[image_output, summary_output],
        api_name="segment",
    )


if __name__ == "__main__":
    demo.launch(mcp_server=True)
