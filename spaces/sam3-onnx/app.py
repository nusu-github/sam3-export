"""ZeroGPU demo for the released SAM3 M2--M4 ONNX Runtime bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

# ZeroGPU must install its CUDA hooks before importing anything CUDA-aware.
import spaces  # isort: skip

import gradio as gr
from huggingface_hub import snapshot_download
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

MODEL_ID = os.environ.get("SAM3_ONNX_MODEL_ID", "RamRom/sam3-onnx")
MODEL_REVISION = os.environ.get("SAM3_ONNX_MODEL_REVISION", "release-m2-m4-ortcuda-v1")
MAX_VIDEO_FRAMES = 60
MAX_OBJECTS = 5

PCS_PLAN = "sam3_base_image_pcs_text_ortcuda_v1"
INTERACTIVE_PLAN = "sam3_base_interactive_image_pvs_ortcuda_v1"
VIDEO_PLAN = "sam3_base_video_tracking_ortcuda_v1"


def _bundle_root() -> Path:
    """Download the tagged, manifest-driven M2--M4 release once per Space build."""
    local_dir = os.environ.get("SAM3_ONNX_MODEL_DIR")
    if local_dir:
        return Path(local_dir)
    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
    )


_MODEL_ROOT = _bundle_root()


def _link_resolved(source: str, destination: str) -> str:
    """Hard-link a Hub snapshot's resolved blob rather than its symlink entry."""
    os.link(os.path.realpath(source), destination)
    return destination


def _bundle(name: str) -> Path:
    source = _MODEL_ROOT / "bundles" / name
    if not source.is_dir():
        raise RuntimeError(f"Released bundle is missing: {source}")

    # Hub's snapshot cache may retain this byte-identical bundle-local LICENSE
    # as a dangling deduplicated symlink.  Make a writable runtime view once:
    # preserve the manifest's license from the Space source and hard-link every
    # resolved graph/artifact blob without copying model bytes.
    runtime = Path.home() / ".cache" / "sam3-onnx-runtime" / f"{name}-v3"
    marker = runtime / ".ready"
    if not marker.is_file():
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).with_name("LICENSE"), runtime / "LICENSE")
        shutil.copytree(
            source,
            runtime,
            copy_function=_link_resolved,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("LICENSE"),
        )
        marker.touch()
    return runtime


def _as_pil(image: Image.Image | np.ndarray | None) -> Image.Image:
    if image is None:
        raise gr.Error("Upload an image first.")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _parse_points(points_json: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Parse [[x, y, label], ...], where label is 1 foreground or 0 background."""
    if not points_json.strip():
        return None, None
    try:
        values = json.loads(points_json)
    except json.JSONDecodeError as exc:
        raise gr.Error(
            "Points must be JSON such as [[120, 80, 1], [160, 90, 0]]."
        ) from exc
    if not isinstance(values, list) or len(values) > 16:
        raise gr.Error("Provide between 0 and 16 points.")
    coordinates: list[list[float]] = []
    labels: list[int] = []
    for point in values:
        if not isinstance(point, list) or len(point) != 3 or point[2] not in (0, 1):
            raise gr.Error("Each point must be [x, y, label], with label 0 or 1.")
        coordinates.append([float(point[0]), float(point[1])])
        labels.append(int(point[2]))
    return np.asarray(coordinates, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _parse_box(box_json: str) -> tuple[float, float, float, float] | None:
    if not box_json.strip():
        return None
    try:
        values = json.loads(box_json)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError
        return tuple(float(value) for value in values)  # type: ignore[return-value]
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise gr.Error("Box must be JSON [left, top, right, bottom].") from exc


def _prompt(points_json: str, box_json: str, mask_logits: Any = None) -> Any:
    """Build the public M3/M4 prompt; raw binary upload masks are not accepted."""
    from sam3.runtime import InteractivePrompt

    points, labels = _parse_points(points_json)
    logits = None if mask_logits is None else np.asarray(mask_logits, dtype=np.float32)
    return InteractivePrompt(
        points_xy=points,
        point_labels=labels,
        box_xyxy=_parse_box(box_json),
        mask_logits=logits,
    )


def _overlay(image: Image.Image, masks: np.ndarray, scores: np.ndarray) -> Image.Image:
    canvas = image.convert("RGBA")
    palette = [
        (46, 196, 182),
        (255, 159, 67),
        (84, 160, 255),
        (255, 107, 107),
        (168, 85, 247),
    ]
    for index, (mask, score) in enumerate(zip(masks, scores)):
        color = palette[index % len(palette)]
        alpha = Image.fromarray((mask.astype(np.uint8) * 110)).resize(
            canvas.size, Image.Resampling.NEAREST
        )
        layer = Image.new("RGBA", canvas.size, (*color, 0))
        layer.putalpha(alpha)
        canvas.alpha_composite(layer)
        ImageDraw.Draw(canvas).text(
            (8, 8 + index * 20), f"#{index + 1}: {score:.3f}", fill=(*color, 255)
        )
    return canvas.convert("RGB")


@spaces.GPU(duration=90)
def segment_text(
    image: Image.Image, concept: str, threshold: float, nms_iou: float, max_results: int
) -> tuple[Image.Image, str]:
    """Run the released M2 text-only image PCS ONNX plan on one image."""
    from sam3.runtime import PredictOptions, create_image_session

    if not concept.strip():
        raise gr.Error("Enter a text concept, such as 'dog' or 'red car'.")
    session = create_image_session(
        PCS_PLAN, bundle_dir=_bundle("sam3-image-pcs-ortcuda-v2")
    )
    try:
        pil = _as_pil(image)
        session.set_image(pil)
        session.set_text(concept.strip())
        prediction = session.predict_text(
            PredictOptions(
                score_threshold=threshold,
                nms_iou_threshold=nms_iou,
                max_results=int(max_results),
            )
        )
        return _overlay(
            pil, prediction.masks >= 0.5, prediction.scores
        ), f"**{len(prediction.scores)} match(es)** — plan `{PCS_PLAN}`"
    finally:
        session.close()


@spaces.GPU(duration=90)
def interactive_predict(
    image: Image.Image,
    points_json: str,
    box_json: str,
    use_prior: bool,
    prior_logits: Any,
    single_mask: bool,
) -> tuple[Image.Image, str, Any]:
    """Preview M3 point/box/prior-logit interactive segmentation on one image."""
    from sam3.runtime import InteractivePredictOptions, create_interactive_session

    pil = _as_pil(image)
    prompt = _prompt(points_json, box_json, prior_logits if use_prior else None)
    session = create_interactive_session(
        INTERACTIVE_PLAN, bundle_dir=_bundle("sam3-interactive-image-pvs-ortcuda-v2")
    )
    try:
        session.set_image(pil)
        result = session.predict(
            prompt, InteractivePredictOptions(multimask_output=not single_mask)
        )
        return (
            _overlay(pil, result.masks, result.scores),
            f"**{len(result.scores)} candidate(s)** — plan `{INTERACTIVE_PLAN}`",
            result.low_res_logits.tolist(),
        )
    finally:
        session.close()


def _load_video(path: str) -> tuple[list[Image.Image], float]:
    if not path:
        raise gr.Error("Upload a video first.")
    frames = iio.imread(path, plugin="pyav")
    if frames.ndim != 4:
        raise gr.Error("The uploaded file did not contain video frames.")
    indices = np.linspace(
        0, len(frames) - 1, min(len(frames), MAX_VIDEO_FRAMES), dtype=np.int64
    )
    selected = [Image.fromarray(frames[index]).convert("RGB") for index in indices]
    metadata = iio.immeta(path, plugin="pyav")
    fps = float(metadata.get("fps", 12.0))
    return selected, min(max(fps, 1.0), 12.0)


def _history_prompt(item: dict[str, Any]) -> Any:
    return _prompt(item["points"], item["box"], item.get("mask_logits"))


def _replay_conditions(session: Any, history: list[dict[str, Any]]) -> None:
    from sam3.runtime import InteractivePredictOptions

    for item in history:
        session.add_object(int(item["object_id"]))
        preview = session.preview(
            int(item["object_id"]),
            int(item["frame_index"]),
            _history_prompt(item),
            InteractivePredictOptions(multimask_output=False),
        )
        session.commit(preview.preview_handle)


def _video_frame_preview(
    frames: list[Image.Image], frame_index: int, masks: np.ndarray, scores: np.ndarray
) -> Image.Image:
    return _overlay(frames[frame_index], masks, scores)


@spaces.GPU(duration=180)
def preview_video(
    video: str,
    object_id: int,
    frame_index: int,
    points_json: str,
    box_json: str,
    history: list[dict[str, Any]],
) -> tuple[Image.Image, str, Any]:
    """Preview three M4 masks for a new point/box correction without mutating tracker state."""
    from sam3.runtime import InteractivePredictOptions, create_video_session

    frames, _ = _load_video(video)
    if not 0 <= int(frame_index) < len(frames):
        raise gr.Error(f"Frame index must be between 0 and {len(frames) - 1}.")
    session = create_video_session(
        VIDEO_PLAN, bundle_dir=_bundle("sam3-base-video-tracking-ortcuda-v2")
    )
    try:
        session.set_video(frames)
        _replay_conditions(session, history or [])
        if int(object_id) not in {int(item["object_id"]) for item in history or []}:
            session.add_object(int(object_id))
        result = session.preview(
            int(object_id),
            int(frame_index),
            _prompt(points_json, box_json),
            InteractivePredictOptions(multimask_output=True),
        )
        return (
            _video_frame_preview(frames, int(frame_index), result.masks, result.scores),
            "Choose a candidate, then commit it to propagate.",
            result.low_res_logits.tolist(),
        )
    finally:
        session.close()


@spaces.GPU(duration=180)
def commit_and_track(
    video: str,
    object_id: int,
    frame_index: int,
    points_json: str,
    box_json: str,
    candidate_index: int,
    preview_logits: Any,
    history: list[dict[str, Any]],
) -> tuple[str, Image.Image, str, list[dict[str, Any]]]:
    """Commit one M4 single-mask correction and propagate all active objects to the final frame."""
    from sam3.runtime import InteractivePredictOptions, create_video_session

    frames, fps = _load_video(video)
    if not preview_logits:
        raise gr.Error("Preview masks first, then choose a candidate to commit.")
    candidates = np.asarray(preview_logits, dtype=np.float32)
    if candidates.ndim != 3 or not 0 <= int(candidate_index) < len(candidates):
        raise gr.Error("Candidate index must select one displayed preview mask.")
    history = list(history or [])
    session = create_video_session(
        VIDEO_PLAN, bundle_dir=_bundle("sam3-base-video-tracking-ortcuda-v2")
    )
    try:
        session.set_video(frames)
        _replay_conditions(session, history)
        object_ids = {int(item["object_id"]) for item in history}
        if int(object_id) not in object_ids:
            if len(object_ids) >= MAX_OBJECTS:
                raise gr.Error(f"This demo supports at most {MAX_OBJECTS} objects.")
            session.add_object(int(object_id))
        entry = {
            "object_id": int(object_id),
            "frame_index": int(frame_index),
            "points": points_json,
            "box": box_json,
            "mask_logits": candidates[int(candidate_index)].tolist(),
        }
        single = session.preview(
            int(object_id),
            int(frame_index),
            _history_prompt(entry),
            InteractivePredictOptions(multimask_output=False),
        )
        session.commit(single.preview_handle)
        history.append(entry)
        predictions = session.propagate(
            start_frame=int(frame_index), end_frame=len(frames) - 1
        )
        rendered: list[np.ndarray] = []
        for output in predictions:
            rendered.append(
                np.asarray(
                    _overlay(frames[output.frame_index], output.masks, output.scores)
                )
            )
        output_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
        iio.imwrite(output_path, np.stack(rendered), fps=fps, plugin="pyav")
        final = Image.fromarray(rendered[-1])
        return (
            str(output_path),
            final,
            f"Committed object {object_id} at frame {frame_index}; propagated {len(predictions)} frame(s).",
            history,
        )
    finally:
        session.close()


with gr.Blocks(theme=gr.themes.Soft(), title="SAM3 ONNX Runtime — M2 to M4") as demo:
    gr.Markdown(
        "# SAM3 ONNX Runtime: M2–M4\n"
        "CUDA ONNX Runtime + IOBinding release bundles. Supports text image PCS, interactive image PVS, and base-video tracking. "
        "It excludes SAM3.1 Multiplex, streaming video, and CPU fallback."
    )
    with gr.Tab("Text image PCS (M2)"):
        with gr.Row():
            with gr.Column():
                pcs_image = gr.Image(
                    type="pil", label="Image", sources=["upload", "clipboard"]
                )
                pcs_text = gr.Textbox(
                    label="Concept", placeholder="e.g. dog, person, red car"
                )
                pcs_threshold = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Score threshold"
                )
                pcs_nms = gr.Slider(
                    0.0,
                    1.0,
                    value=1.0,
                    step=0.05,
                    label="Mask NMS IoU (1 disables suppression)",
                )
                pcs_max = gr.Slider(1, 200, value=25, step=1, label="Maximum results")
                pcs_run = gr.Button("Segment", variant="primary")
            with gr.Column():
                pcs_output = gr.Image(type="pil", label="Segmentation")
                pcs_summary = gr.Markdown()
        pcs_run.click(
            segment_text,
            [pcs_image, pcs_text, pcs_threshold, pcs_nms, pcs_max],
            [pcs_output, pcs_summary],
            api_name="segment_text",
        )

    with gr.Tab("Interactive image (M3)"):
        with gr.Row():
            with gr.Column():
                interactive_image = gr.Image(
                    type="pil", label="Image", sources=["upload", "clipboard"]
                )
                interactive_points = gr.Textbox(
                    label="Points JSON", placeholder="[[120, 80, 1], [160, 90, 0]]"
                )
                interactive_box = gr.Textbox(
                    label="Box JSON (optional)",
                    placeholder="[left, top, right, bottom]",
                )
                prior_toggle = gr.Checkbox(
                    label="Use selected prior low-resolution mask", value=False
                )
                single_toggle = gr.Checkbox(label="Single-mask output", value=False)
                interactive_run = gr.Button("Predict", variant="primary")
            with gr.Column():
                interactive_output = gr.Image(type="pil", label="Segmentation")
                interactive_summary = gr.Markdown()
        interactive_logits = gr.State([])
        interactive_run.click(
            interactive_predict,
            [
                interactive_image,
                interactive_points,
                interactive_box,
                prior_toggle,
                interactive_logits,
                single_toggle,
            ],
            [interactive_output, interactive_summary, interactive_logits],
            api_name="interactive_predict",
        )

    with gr.Tab("Video tracking (M4)"):
        gr.Markdown(
            "Upload a video, preview a point/box prompt, select one preview mask, then commit and propagate. Videos are sampled to at most 60 frames; up to 5 objects are supported."
        )
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Video")
                video_object = gr.Number(value=1, precision=0, label="Object ID")
                video_frame = gr.Number(
                    value=0, precision=0, label="Correction frame index"
                )
                video_points = gr.Textbox(
                    label="Points JSON", placeholder="[[120, 80, 1], [160, 90, 0]]"
                )
                video_box = gr.Textbox(
                    label="Box JSON (optional)",
                    placeholder="[left, top, right, bottom]",
                )
                preview_button = gr.Button("Preview masks", variant="secondary")
                candidate = gr.Slider(0, 2, value=0, step=1, label="Preview candidate")
                commit_button = gr.Button("Commit and track", variant="primary")
            with gr.Column():
                preview_image = gr.Image(type="pil", label="Preview")
                video_output = gr.Video(label="Tracked video")
                final_frame = gr.Image(type="pil", label="Final propagated frame")
                video_summary = gr.Markdown()
        video_history = gr.State([])
        video_preview_logits = gr.State([])
        preview_button.click(
            preview_video,
            [
                video_input,
                video_object,
                video_frame,
                video_points,
                video_box,
                video_history,
            ],
            [preview_image, video_summary, video_preview_logits],
            api_name="preview_video",
        )
        commit_button.click(
            commit_and_track,
            [
                video_input,
                video_object,
                video_frame,
                video_points,
                video_box,
                candidate,
                video_preview_logits,
                video_history,
            ],
            [video_output, final_frame, video_summary, video_history],
            api_name="commit_and_track",
        )


if __name__ == "__main__":
    demo.launch(mcp_server=True)
