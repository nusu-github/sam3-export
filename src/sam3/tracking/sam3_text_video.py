"""Text-on-video runtime: ground objects, seed tracks, then propagate.

Simplified single-GPU path inspired by ``Sam3VideoInference.add_prompt`` +
``propagate``, without MultiGPU / hotstart delay buffer.

Pipeline
--------
1. ``init_state(video_path)`` — load frames
2. ``add_text_prompt(text)`` — run detector on ``frame_idx``; seed one tracker
   memory bank per detection (mask prompt)
3. ``propagate()`` — for later frames: tracker propagate; every
   ``recondition_every_nth_frame`` re-run detector + associate + spawn new
   tracks / keep existing

Objects are tracked independently (one ``Sam3VideoTracker`` state per object)
for clarity; quality matches the official per-object SAM2 path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any

from jaxtyping import Bool, Float
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert

from sam3.dtype_policy import PrecisionConfig, module_param_dtype

from ..grounding.sam3_image import FindStage, Sam3Image
from ..runtime.associate_det_trk import associate_det_trk
from ..runtime.video_io import load_video_frames_from_jpg
from .multi_object_propagate import propagate_objects_one_frame
from .sam3_video_tracker import Sam3VideoTracker
from .td_features import unpack_frame_features


@dataclass
class TrackObject:
    obj_id: int
    score: float  # first-frame detection score
    state: dict[str, Any]  # Sam3VideoTracker inference state
    birth_frame: int = 0
    # last low-res mask logits (1,1,h,w) for association
    last_mask: Tensor | None = None


@dataclass
class TextOnVideoState:
    images: Tensor  # (T, 3, S, S)
    video_height: int
    video_width: int
    num_frames: int
    device: torch.device
    text_prompt: str | None = None
    objects: list[TrackObject] = field(default_factory=list)
    next_obj_id: int = 1
    # frame_idx -> {obj_id: binary mask at video res}
    outputs: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)  # obj_id -> score
    shared_vision_features: dict = field(default_factory=dict)
    memmap_dir: str | Path | None = None
    memmap_vision: bool = True
    memmap_outputs: bool = True


class Sam3TextOnVideo(nn.Module):
    """Single-GPU text open-vocab video segmentation."""

    def __init__(
        self,
        detector: Sam3Image,
        tracker: nn.Module,
        *,
        confidence_threshold: float | int = 0.5,
        assoc_iou_thresh: float | int = 0.1,
        trk_assoc_iou_thresh: float | int = 0.5,
        new_det_thresh: float | int = 0.7,
        recondition_every_nth_frame: int = 16,
        image_size: int = 1008,
        precision: PrecisionConfig | None = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        self.precision = precision
        self.video_tracker = Sam3VideoTracker(tracker, precision=precision)
        self.confidence_threshold = float(confidence_threshold)
        self.assoc_iou_thresh = float(assoc_iou_thresh)
        self.trk_assoc_iou_thresh = float(trk_assoc_iou_thresh)
        self.new_det_thresh = float(new_det_thresh)
        self.recondition_every_nth_frame = int(recondition_every_nth_frame)
        self.image_size = int(image_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _compute_dtype(self) -> torch.dtype:
        if self.precision is not None:
            return self.precision.compute_dtype
        try:
            return module_param_dtype(self)
        except Exception:
            return next(self.parameters()).dtype

    # ----------------------------------------------------------------- state

    @torch.inference_mode()
    def init_state(
        self,
        video_path: str,
        *,
        max_frames: int | None = None,
        offload_video_to_cpu: bool = False,
        device: torch.device | None = None,
        memmap_dir: str | Path | None = None,
        memmap_vision: bool = True,
        memmap_outputs: bool = True,
    ) -> TextOnVideoState:
        if device is None:
            device = self.device
        images, vh, vw = load_video_frames_from_jpg(
            video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            max_frames=max_frames,
            device=device if not offload_video_to_cpu else "cpu",
        )
        return TextOnVideoState(
            images=images,
            video_height=vh,
            video_width=vw,
            num_frames=int(images.shape[0]),
            device=device,
            memmap_dir=Path(memmap_dir).expanduser()
            if memmap_dir is not None
            else None,
            memmap_vision=memmap_dir is not None and memmap_vision,
            memmap_outputs=memmap_dir is not None and memmap_outputs,
        )

    # --------------------------------------------------------------- detect

    def _detect_frame(
        self, state: TextOnVideoState, frame_idx: int, text: str
    ) -> tuple[Bool[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]:
        """Run text detector on one frame.

        Returns:
            masks_video: ``(N, H, W)`` bool at original video resolution
            boxes_xyxy: ``(N, 4)`` pixel boxes
            scores: ``(N,)``
        """
        dtype = self._compute_dtype()
        img = state.images[frame_idx].to(device=state.device, dtype=dtype).unsqueeze(0)

        # Vision + text through detector backbone
        backbone_out = self.detector.backbone.forward_image(img)
        text_out = self.detector.backbone.forward_text([text], device=state.device)
        backbone_out.update(text_out)
        geometric_prompt = self.detector._get_dummy_prompt(num_prompts=1)
        find_input = FindStage(
            img_ids=torch.tensor([0], device=state.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=state.device, dtype=torch.long),
        )
        outputs = self.detector.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        # pred_logits: (B, Q, 1) or (B, Q); masks: (B, Q, h, w); boxes: (B, Q, 4)
        logits = outputs["pred_logits"].float().sigmoid()
        if logits.dim() == 3:
            logits = logits.squeeze(-1)  # (B, Q)
        presence = outputs["presence_logit_dec"].float().sigmoid()  # (B, 1) or (B,)
        while presence.dim() < 2:
            presence = presence.unsqueeze(-1)
        # broadcast presence over queries: (B, 1) * (B, Q)
        probs = (logits * presence).reshape(-1)  # (Q,) for B=1
        masks_lr = outputs["pred_masks"]
        boxes = outputs["pred_boxes"]
        if masks_lr.dim() == 4:
            masks_lr = masks_lr[0]  # (Q, h, w)
        if boxes.dim() == 3:
            boxes = boxes[0]  # (Q, 4)

        keep = probs > self.confidence_threshold
        if not keep.any():
            empty = torch.zeros(
                0, state.video_height, state.video_width, device=state.device
            )
            return (
                empty.bool(),
                torch.zeros(0, 4, device=state.device),
                torch.zeros(0, device=state.device),
            )

        probs = probs[keep]
        masks_lr = masks_lr[keep]
        boxes = boxes[keep]

        # Sort by score desc
        order = torch.argsort(probs, descending=True)
        probs = probs[order]
        masks_lr = masks_lr[order]
        boxes = boxes[order]

        masks_v = F.interpolate(
            masks_lr.unsqueeze(1).float(),
            size=(state.video_height, state.video_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        masks_bool = masks_v > 0.0

        # boxes to pixel xyxy
        boxes_xyxy = box_convert(
            boxes.reshape(-1, 4), in_fmt="cxcywh", out_fmt="xyxy"
        ).reshape_as(boxes)
        scale = torch.tensor(
            [
                state.video_width,
                state.video_height,
                state.video_width,
                state.video_height,
            ],
            device=boxes.device,
            dtype=boxes_xyxy.dtype,
        )
        boxes_xyxy = boxes_xyxy * scale
        return masks_bool, boxes_xyxy, probs

    # ---------------------------------------------------------- seed / track

    def _seed_object_from_mask(
        self,
        state: TextOnVideoState,
        frame_idx: int,
        mask_video: Tensor,
        score: float | int,
    ) -> TrackObject:
        """Create a new track from a high-res mask on ``frame_idx``."""
        # Tracker wants mask at model input resolution as soft logits
        dtype = self._compute_dtype()
        mask_model = F.interpolate(
            mask_video[None, None].float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # map {0,1} → approx logits; keep model weight dtype for mask_downsample
        mask_model = (mask_model * 20.0 - 10.0).to(device=state.device, dtype=dtype)

        # Fresh video-tracker state sharing the same images
        init_kwargs = dict(
            images=state.images,
            video_height=state.video_height,
            video_width=state.video_width,
            device=state.device,
            memmap_dir=state.memmap_dir,
            memmap_vision=state.memmap_vision,
            memmap_outputs=state.memmap_outputs,
        )
        init_sig = inspect.signature(self.video_tracker.init_state)
        if "shared_vision_features" in init_sig.parameters:
            vt_state = self.video_tracker.init_state(
                shared_vision_features=state.shared_vision_features, **init_kwargs
            )
        else:
            vt_state = self.video_tracker.init_state(**init_kwargs)
            vt_state["vision_features"] = state.shared_vision_features
        # Manually inject mask track_step as conditioning frame
        vision_feats, vision_pos_embeds, feat_sizes, image = unpack_frame_features(
            self.video_tracker._get_frame_features(vt_state, frame_idx),
            device=state.device,
        )
        current_out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=None,
            mask_inputs=mask_model,
            output_dict=vt_state["output_dict"],
            num_frames=state.num_frames,
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
            use_prev_mem_frame=True,
        )
        self.video_tracker._store_frame_output(
            vt_state, "cond_frame_outputs", frame_idx, current_out
        )
        vt_state["frames_already_tracked"][frame_idx] = {"reverse": False}
        vt_state["first_ann_frame_idx"] = frame_idx

        obj_id = state.next_obj_id
        state.next_obj_id += 1
        state.scores[obj_id] = float(score)
        return TrackObject(
            obj_id=obj_id,
            score=float(score),
            birth_frame=frame_idx,
            state=vt_state,
            last_mask=current_out["pred_masks"].detach(),
        )

    def _store_frame_output(
        self, state: TextOnVideoState, frame_idx: int, obj: TrackObject
    ) -> None:
        out = obj.state["output_dict"]["cond_frame_outputs"].get(frame_idx)
        if out is None:
            out = obj.state["output_dict"]["non_cond_frame_outputs"].get(frame_idx)
        if out is None or out.get("pred_masks") is None:
            return
        masks = out["pred_masks"]  # (1,1,h,w)
        m = F.interpolate(
            masks.float(),
            size=(state.video_height, state.video_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        binary = m > 0.0
        if frame_idx not in state.outputs:
            state.outputs[frame_idx] = {}
        state.outputs[frame_idx][obj.obj_id] = binary
        obj.last_mask = masks.detach()

    # ------------------------------------------------------------- public API

    @torch.inference_mode()
    def add_text_prompt(
        self,
        state: TextOnVideoState,
        text: str,
        *,
        frame_idx: int = 0,
    ) -> dict[str, Any]:
        """Detect ``text`` on ``frame_idx`` and seed tracks."""
        state.text_prompt = text
        state.objects.clear()
        state.outputs.clear()
        state.scores.clear()
        state.next_obj_id = 1

        masks, boxes, scores = self._detect_frame(state, frame_idx, text)
        for i in range(masks.shape[0]):
            obj = self._seed_object_from_mask(
                state, frame_idx, masks[i], float(scores[i])
            )
            state.objects.append(obj)
            self._store_frame_output(state, frame_idx, obj)

        return self._pack_frame(state, frame_idx)

    @torch.inference_mode()
    def propagate(
        self,
        state: TextOnVideoState,
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        recondition: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Propagate tracks across frames; optionally re-detect + associate."""
        if state.text_prompt is None:
            raise RuntimeError("call add_text_prompt first")
        if start_frame_idx is None:
            start_frame_idx = 0
        if max_frame_num_to_track is None:
            max_frame_num_to_track = state.num_frames
        end = min(start_frame_idx + max_frame_num_to_track, state.num_frames)
        do_recond = (
            self.recondition_every_nth_frame > 0 if recondition is None else recondition
        )

        results: list[dict[str, Any]] = []
        for frame_idx in range(start_frame_idx, end):
            if frame_idx == start_frame_idx and frame_idx in state.outputs:
                # already seeded
                results.append(self._pack_frame(state, frame_idx))
                continue

            current_outs = propagate_objects_one_frame(
                self.video_tracker,
                [obj.state for obj in state.objects],
                frame_idx,
                shared_vision_cache=state.shared_vision_features,
                run_mem_encoder=True,
            )
            for obj, out in zip(state.objects, current_outs):
                if out is not None:
                    self._store_frame_output(state, frame_idx, obj)

            should_recondition = False
            if do_recond:
                should_recondition = (
                    frame_idx > start_frame_idx
                    and frame_idx % self.recondition_every_nth_frame == 0
                )
            if should_recondition and state.text_prompt is not None:
                self._recondition_frame(state, frame_idx)

            results.append(self._pack_frame(state, frame_idx))
        return results

    def _recondition_frame(self, state: TextOnVideoState, frame_idx: int) -> None:
        """Re-run detector and spawn tracks for unmatched high-score dets."""
        masks, boxes, scores = self._detect_frame(state, frame_idx, state.text_prompt)
        if masks.numel() == 0:
            return

        # Collect current track masks at video res
        trk_masks = []
        trk_ids = []
        for obj in state.objects:
            frame_out = state.outputs.get(frame_idx, {}).get(obj.obj_id)
            if frame_out is None:
                continue
            trk_masks.append(frame_out.float())
            trk_ids.append(obj.obj_id)

        if trk_masks:
            trk_stack = torch.stack(trk_masks, dim=0)
            new_inds, unmatched, det_to_trk, _ = associate_det_trk(
                det_masks=masks.float(),
                track_masks=trk_stack,
                iou_threshold=self.assoc_iou_thresh,
                iou_threshold_trk=self.trk_assoc_iou_thresh,
                det_scores=scores,
                new_det_thresh=self.new_det_thresh,
            )
        else:
            new_inds = list(range(masks.shape[0]))
            # filter by new_det_thresh
            new_inds = [i for i in new_inds if float(scores[i]) >= self.new_det_thresh]

        for i in new_inds:
            obj = self._seed_object_from_mask(
                state, frame_idx, masks[i], float(scores[i])
            )
            state.objects.append(obj)
            self._store_frame_output(state, frame_idx, obj)

    def _pack_frame(self, state: TextOnVideoState, frame_idx: int) -> dict[str, Any]:
        objs = state.outputs.get(frame_idx, {})
        obj_ids = sorted(objs.keys())
        if not obj_ids:
            return {
                "frame_idx": frame_idx,
                "obj_ids": [],
                "masks": torch.zeros(
                    0, state.video_height, state.video_width, dtype=torch.bool
                ),
                "scores": torch.zeros(0),
            }
        masks = torch.stack([objs[i] for i in obj_ids], dim=0)
        scores = torch.tensor(
            [state.scores.get(i, 0.0) for i in obj_ids], dtype=torch.float32
        )
        return {
            "frame_idx": frame_idx,
            "obj_ids": obj_ids,
            "masks": masks,
            "scores": scores,
        }
