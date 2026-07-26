"""Multi-frame runtime wrapper around the per-object tracker.

Wraps ``Sam3Tracker`` and manages conditioning/non-conditioning frame state
across a clip.

This is intentionally **not** the MultiGPU request server, hotstart path, or
detector-association video predictor — just ``init_state`` → ``add_points`` →
``propagate`` over ``track_step``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jaxtyping import Float, Integer
from tensordict import TensorDictBase
import torch
from torch import Tensor
import torch.nn as nn

from sam3.dtype_policy import PrecisionConfig, module_param_dtype

from .sam3_tracker import Sam3Tracker
from .td_features import (
    as_frame_features,
    materialize_td,
    memmap_frame_out,
    memmap_td,
    unpack_frame_features,
)

PointInputs = dict[str, Tensor]
FrameFeatures = TensorDictBase
InferenceState = dict[str, Any]


def _as_batch_points(
    points: Float[Tensor, "..."] | Sequence[Sequence[float | int]],
    labels: Integer[Tensor, "..."] | Sequence[int],
    device: torch.device,
) -> PointInputs:
    """Normalize points/labels to ``{point_coords: (1,N,2), point_labels: (1,N)}``."""
    if not isinstance(points, Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    if not isinstance(labels, Tensor):
        labels = torch.tensor(labels, dtype=torch.int32)
    points = points.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.int32)
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if points.shape[:2] != labels.shape[:2]:
        raise ValueError(
            f"points batch shape {tuple(points.shape[:2])} != "
            f"labels batch shape {tuple(labels.shape[:2])}"
        )
    return {"point_coords": points, "point_labels": labels}


def concat_points(
    old: Mapping[str, Tensor] | None,
    new_points: Float[Tensor, "..."],
    new_labels: Integer[Tensor, "..."] | Float[Tensor, "..."],
) -> PointInputs:
    """Append points to existing prompt inputs (same as official tracker util)."""
    if old is None:
        return {"point_coords": new_points, "point_labels": new_labels}
    return {
        "point_coords": torch.cat([old["point_coords"], new_points], dim=1),
        "point_labels": torch.cat([old["point_labels"], new_labels], dim=1),
    }


class Sam3VideoTracker(nn.Module):
    """SAM2-style multi-frame shell around a ``Sam3Tracker`` core.

    Parameters
    ----------
    tracker:
        ``Sam3Tracker`` (or compatible). Must implement
        ``track_step(...)`` matching official ``Sam3TrackerBase.track_step``.
    image_size:
        Model input resolution (default 1008 for production SAM3 tracker).
        Used when normalizing relative point coords.
    """

    def __init__(
        self,
        tracker: nn.Module,
        *,
        image_size: int | None = None,
        precision: PrecisionConfig | None = None,
    ) -> None:
        super().__init__()
        self.tracker = tracker
        self.precision = precision
        if image_size is not None:
            self.image_size = int(image_size)
        else:
            self.image_size = int(getattr(tracker, "image_size", 1008))

    def _compute_dtype(self) -> torch.dtype:
        if self.precision is not None:
            return self.precision.compute_dtype
        try:
            return module_param_dtype(self.tracker)
        except Exception:
            try:
                return module_param_dtype(self)
            except Exception:
                return next(self.parameters()).dtype

    def _to_storage(self, tensor: Any) -> Any:
        if self.precision is None:
            return tensor
        storage_dtype = self.precision.resolved_storage()
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if not torch.is_floating_point(tensor):
            return tensor
        return tensor.to(dtype=storage_dtype)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.tracker.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------ state

    @torch.inference_mode()
    def init_state(
        self,
        *,
        num_frames: int | None = None,
        video_height: int | None = None,
        video_width: int | None = None,
        vision_features: Sequence[dict[str, Any] | FrameFeatures] | None = None,
        video_path: str | None = None,
        images: Float[Tensor, "t 3 h w"] | None = None,
        max_frames: int | None = None,
        offload_video_to_cpu: bool = False,
        device: torch.device | None = None,
        precompute_features: bool = False,
        shared_vision_features: dict[int, dict[str, Any] | FrameFeatures] | None = None,
        memmap_dir: str | Path | None = None,
        memmap_vision: bool = True,
        memmap_outputs: bool = True,
    ) -> InferenceState:
        """Create multi-frame inference state.

        Provide **one** of:
          * ``vision_features`` — precomputed feats (unit tests)
          * ``images`` — tensor ``(T, 3, S, S)`` already normalized
          * ``video_path`` — JPEG folder (loaded via ``video_io``)
          * ``num_frames`` alone — empty state; encode lazily if backbone set
        ``shared_vision_features`` — optional pre-allocated feature cache shared
        across states (same object identity).
        """
        if device is None:
            device = self.device
        if memmap_dir is not None:
            storage_device = torch.device("cpu")
        else:
            storage_device = device

        loaded_images: Tensor | None = None
        if video_path is not None:
            from ..runtime.video_io import load_video_frames_from_jpg

            loaded_images, vh, vw = load_video_frames_from_jpg(
                video_path,
                image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                max_frames=max_frames,
                device=device if not offload_video_to_cpu else "cpu",
            )
            video_height = vh
            video_width = vw
            num_frames = loaded_images.shape[0]
        elif images is not None:
            loaded_images = images
            num_frames = int(images.shape[0])
            if video_height is None:
                video_height = self.image_size
            if video_width is None:
                video_width = self.image_size
        elif vision_features is not None:
            num_frames = len(vision_features)
        if num_frames is None or num_frames < 1:
            raise ValueError(
                "init_state requires vision_features, images, video_path, or num_frames"
            )

        if video_height is None:
            video_height = self.image_size
        if video_width is None:
            video_width = self.image_size
        cached: dict[int, dict[str, Any] | FrameFeatures] = (
            {} if shared_vision_features is None else shared_vision_features
        )
        if memmap_dir is not None:
            memmap_root = Path(memmap_dir).expanduser()
        else:
            memmap_root = None
        if vision_features is not None:
            for t, feats in enumerate(vision_features):
                prefix = None
                if memmap_root is not None:
                    prefix = memmap_root / "vision" / f"{t:06d}"
                cached[t] = self._normalize_frame_features(
                    feats, device, memmap_prefix=prefix
                )

        state: InferenceState = {
            "num_frames": int(num_frames),
            "video_height": int(video_height),
            "video_width": int(video_width),
            "device": device,
            "storage_device": storage_device,
            "memmap_dir": memmap_root,
            "memmap_vision": bool(memmap_dir is not None and memmap_vision),
            "memmap_outputs": bool(memmap_dir is not None and memmap_outputs),
            "images": loaded_images,
            "vision_features": cached,
            "point_inputs": {},
            "mask_inputs": {},
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "frames_already_tracked": {},
            "_sam3_video_tracker": self,
            "tracking_has_started": False,
            "first_ann_frame_idx": None,
        }
        state["memmap_state_id"] = id(state)

        if precompute_features and loaded_images is not None:
            for t in range(int(num_frames)):
                self._encode_frame_into_cache(state, t)
        return state

    def _vision_cache_prefix(
        self, state: InferenceState, frame_idx: int
    ) -> Path | None:
        if not state.get("memmap_vision", False):
            return None
        memmap_dir = state.get("memmap_dir")
        if memmap_dir is None:
            return None
        return Path(memmap_dir) / "vision" / f"{frame_idx:06d}"

    def _output_cache_prefix(
        self, state: InferenceState, storage_key: str, frame_idx: int
    ) -> Path | None:
        if not state.get("memmap_outputs", False):
            return None
        memmap_dir = state.get("memmap_dir")
        if memmap_dir is None:
            return None
        state_id = state.get("memmap_state_id", id(state))
        return (
            Path(memmap_dir)
            / "outputs"
            / str(state_id)
            / storage_key
            / f"{frame_idx:06d}"
        )

    def _store_frame_output(
        self,
        state: InferenceState,
        storage_key: str,
        frame_idx: int,
        out: dict[str, Any],
    ) -> None:
        if self.precision is not None:
            if isinstance(out, TensorDictBase):
                if "maskmem_features" in out.keys():
                    out = out.copy()
                    out["maskmem_features"] = self._to_storage(out["maskmem_features"])
            elif isinstance(out, dict):
                mm = out.get("maskmem_features")
                if mm is not None:
                    out = dict(out)
                    out["maskmem_features"] = self._to_storage(mm)
        if state.get("memmap_outputs", False):
            prefix = self._output_cache_prefix(state, storage_key, frame_idx)
            if prefix is not None:
                if isinstance(out, TensorDictBase):
                    out = dict(out)
                out = memmap_frame_out(out, prefix)
        state["output_dict"][storage_key][frame_idx] = out

    def _normalize_frame_features(
        self,
        feats: FrameFeatures,
        device: torch.device,
        memmap_prefix: Path | None = None,
    ) -> FrameFeatures:
        out = materialize_td(as_frame_features(feats), device)
        image = out.get("image")
        if image is None:
            # Dummy image placeholder; track_step may ignore it when features
            # are precomputed. Shape matches image_size if tracker has it.
            level0 = next(iter(out["vision_feats"].values()))
            out["image"] = torch.zeros(
                level0.shape[1], 3, self.image_size, self.image_size, device=device
            )
        if memmap_prefix is None:
            return out
        return memmap_td(out, memmap_prefix)

    def _get_frame_features(
        self, state: InferenceState, frame_idx: int
    ) -> FrameFeatures:
        cached = state["vision_features"]
        if frame_idx in cached:
            feats = cached[frame_idx]
            if not isinstance(feats, TensorDictBase):
                feats = as_frame_features(feats)
                cached[frame_idx] = feats
            return feats
        return self._encode_frame_into_cache(state, frame_idx)

    def _encode_frame_into_cache(
        self, state: InferenceState, frame_idx: int
    ) -> FrameFeatures:
        """Run tracker backbone on one frame and cache flattened features."""
        if getattr(self.tracker, "backbone", None) is None:
            raise RuntimeError(
                f"No precomputed vision features for frame {frame_idx} and "
                "tracker.backbone is None. Pass vision_features/images with "
                "a backbone-attached tracker (with_backbone=True)."
            )
        images = state.get("images")
        if images is None:
            raise RuntimeError(
                f"Cannot encode frame {frame_idx}: state has no images tensor"
            )
        device = state["device"]
        dtype = self._compute_dtype()
        img = images[frame_idx].to(device=device, dtype=dtype).unsqueeze(0)
        backbone_out = self.tracker.forward_image(img)
        _, vision_feats, vision_pos, feat_sizes = (
            self.tracker._prepare_backbone_features(backbone_out)
        )
        feats = {
            "vision_feats": vision_feats,
            "vision_pos_embeds": vision_pos,
            "feat_sizes": feat_sizes,
            "image": img,
        }
        cache_prefix = self._vision_cache_prefix(state, frame_idx)
        normalized = self._normalize_frame_features(
            feats, device, memmap_prefix=cache_prefix
        )
        state["vision_features"][frame_idx] = normalized
        return normalized

    # ------------------------------------------------------------------ points

    @torch.inference_mode()
    def add_points(
        self,
        state: InferenceState,
        frame_idx: int,
        points: Float[Tensor, "..."] | Sequence[Sequence[float | int]],
        labels: Integer[Tensor, "..."] | Sequence[int],
        *,
        clear_old_points: bool = True,
        normalize_coords: bool = False,
        run_mem_encoder: bool = True,
    ) -> dict[str, Any] | TensorDictBase:
        """Add point prompts on ``frame_idx`` and run a conditioning ``track_step``.

        Parameters
        ----------
        normalize_coords:
            If True, treat ``points`` as relative in [0, 1] and scale by
            ``image_size`` (SAM2-demo style).
        run_mem_encoder:
            Encode the predicted mask into the memory bank (default True so
            subsequent ``propagate`` can attend to this frame).
        """
        if not (0 <= frame_idx < state["num_frames"]):
            raise IndexError(
                f"frame_idx={frame_idx} out of range [0, {state['num_frames']})"
            )

        device = state["device"]
        point_inputs = _as_batch_points(points, labels, device)
        if normalize_coords:
            point_inputs["point_coords"] = point_inputs["point_coords"] * float(
                self.image_size
            )

        if clear_old_points:
            merged = point_inputs
        else:
            merged = concat_points(
                state["point_inputs"].get(frame_idx),
                point_inputs["point_coords"],
                point_inputs["point_labels"],
            )
        state["point_inputs"][frame_idx] = merged
        state["mask_inputs"].pop(frame_idx, None)

        is_init_cond = frame_idx not in state["frames_already_tracked"]
        # First-time clicks become conditioning frames.
        is_cond = is_init_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out = self._run_track_step(
            state=state,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond,
            point_inputs=merged,
            mask_inputs=None,
            run_mem_encoder=run_mem_encoder,
            track_in_reverse=False,
        )

        # Drop any prior non-cond entry if this is now a cond frame.
        if is_cond:
            state["output_dict"]["non_cond_frame_outputs"].pop(frame_idx, None)
        self._store_frame_output(state, storage_key, frame_idx, current_out)
        state["frames_already_tracked"][frame_idx] = {"reverse": False}
        if state["first_ann_frame_idx"] is None:
            state["first_ann_frame_idx"] = frame_idx

        return current_out

    # ---------------------------------------------------------------- propagate

    @torch.inference_mode()
    def propagate(
        self,
        state: InferenceState,
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        run_mem_encoder: bool = True,
    ) -> list[tuple[int, dict[str, Any] | TensorDictBase]]:
        """Propagate masks across frames via a ``track_step`` loop.

        Yields are collected into a list of ``(frame_idx, current_out)``.
        Conditioning frames already in ``cond_frame_outputs`` are re-emitted
        without re-running the model (unless only non-cond was stored).
        """
        cond = state["output_dict"]["cond_frame_outputs"]
        if len(cond) == 0:
            raise RuntimeError("No conditioning frames; call add_points first")

        state["tracking_has_started"] = True
        order = self._processing_order(
            state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

        results: list[tuple[int, dict[str, Any] | TensorDictBase]] = []
        for frame_idx in order:
            if frame_idx in cond:
                current_out = cond[frame_idx]
            elif frame_idx in state["output_dict"]["non_cond_frame_outputs"]:
                # Re-emit cached non-cond if present (idempotent re-run).
                current_out = state["output_dict"]["non_cond_frame_outputs"][frame_idx]
            else:
                current_out = self._run_track_step(
                    state=state,
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    run_mem_encoder=run_mem_encoder,
                    track_in_reverse=reverse,
                )
                self._store_frame_output(
                    state, "non_cond_frame_outputs", frame_idx, current_out
                )
                state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            results.append((frame_idx, current_out))
        return results

    def _processing_order(
        self,
        state: InferenceState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> list[int]:
        num_frames = state["num_frames"]
        if start_frame_idx is None:
            start_frame_idx = min(state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                return list(range(start_frame_idx, end - 1, -1))
            return [0]
        end = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        return list(range(start_frame_idx, end + 1))

    # -------------------------------------------------------------- track_step

    def _run_track_step(
        self,
        *,
        state: InferenceState,
        frame_idx: int,
        is_init_cond_frame: bool,
        point_inputs: PointInputs | None,
        mask_inputs: Tensor | None,
        run_mem_encoder: bool,
        track_in_reverse: bool,
    ) -> dict[str, Any] | TensorDictBase:
        vision_feats, vision_pos_embeds, feat_sizes, image = unpack_frame_features(
            self._get_frame_features(state, frame_idx),
            device=state["device"],
        )
        return self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=state["output_dict"],
            num_frames=state["num_frames"],
            track_in_reverse=track_in_reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=None,
            use_prev_mem_frame=True,
        )

    # ---------------------------------------------------------------- helpers

    def reset(self, state: InferenceState) -> None:
        """Clear prompts and tracking outputs (keeps vision feature cache)."""
        state["point_inputs"].clear()
        state["mask_inputs"].clear()
        state["output_dict"]["cond_frame_outputs"].clear()
        state["output_dict"]["non_cond_frame_outputs"].clear()
        state["frames_already_tracked"].clear()
        state["tracking_has_started"] = False
        state["first_ann_frame_idx"] = None


def attach_shared_vision_cache(
    state: InferenceState,
    cache: dict[int, dict[str, Any] | FrameFeatures],
) -> None:
    """Attach a shared vision feature cache to ``state`` by identity."""
    state["vision_features"] = cache


def ensure_frame_features(
    state: InferenceState,
    frame_idx: int,
) -> FrameFeatures:
    """Public helper for fetching/encoding frame features.

    Prefers the owning tracker when available (to preserve encode-once behavior).
    """
    owner = state.get("_sam3_video_tracker")
    if owner is not None and hasattr(owner, "_get_frame_features"):
        return owner._get_frame_features(state, frame_idx)  # type: ignore[call-arg]
    cached = state["vision_features"]
    if frame_idx not in cached:
        raise KeyError(f"Frame {frame_idx} not in vision_features cache")
    feats = cached[frame_idx]
    if not isinstance(feats, TensorDictBase):
        feats = as_frame_features(feats)
        cached[frame_idx] = feats
    return feats


def copy_frame_features(
    src_state: InferenceState,
    dst_state: InferenceState,
    frame_idx: int,
) -> None:
    """Copy a single cached frame entry by reference when available."""
    src_cached = src_state["vision_features"]
    if frame_idx not in src_cached:
        return
    dst_state["vision_features"][frame_idx] = src_cached[frame_idx]


def build_sam3_video_tracker(
    tracker: nn.Module | None = None,
    **tracker_kwargs: Any,
) -> Sam3VideoTracker:
    """Convenience constructor.

    If ``tracker`` is omitted, construct ``Sam3Tracker`` with ``tracker_kwargs``.
    """
    if tracker is None:
        tracker = Sam3Tracker(**tracker_kwargs)
    return Sam3VideoTracker(tracker)
