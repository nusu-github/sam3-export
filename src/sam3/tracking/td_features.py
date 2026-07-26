"""TensorDict helpers for frame-level vision-feature bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tensordict import TensorDict, TensorDictBase
import torch
from torch import Tensor


def _sorted_numeric_keys(levels: Mapping[str | int, Any]) -> list[str | int]:
    numeric_keys: list[tuple[int, str | int]] = []
    for key in levels.keys():
        if isinstance(key, int):
            numeric_keys.append((key, key))
            continue
        if isinstance(key, str):
            try:
                numeric = int(key)
            except ValueError as exc:
                raise TypeError(
                    "nested level keys must be numeric strings/ints"
                ) from exc
            numeric_keys.append((numeric, key))
            continue
        raise TypeError("nested level keys must be numeric strings/ints")

    numeric_keys.sort(key=lambda kv: kv[0])
    return [key for _, key in numeric_keys]


def _levels_list(levels: TensorDictBase | Mapping[str | int, Tensor]) -> list[Tensor]:
    """Convert nested level TensorDict/mapping to ordered level tensors."""
    if not isinstance(levels, (TensorDictBase, Mapping)):
        raise TypeError(f"expected level TensorDict/mapping, got {type(levels)!r}")
    keys = _sorted_numeric_keys(levels)
    values: list[Tensor] = []
    for key in keys:
        value = levels[key]
        if not torch.is_tensor(value):
            raise TypeError(
                f"level value must be tensor, got {type(value)!r} at key {key!r}"
            )
        values.append(value)
    return values


def _levels_td(levels: Sequence[Tensor]) -> TensorDict:
    """Convert per-level tensors into canonical level TensorDict."""
    out: dict[str, Tensor] = {}
    for idx, value in enumerate(list(levels)):
        if not torch.is_tensor(value):
            raise TypeError(
                f"expected tensor level, got {type(value)!r} at index {idx}"
            )
        out[str(idx)] = value
    return TensorDict(out, batch_size=[])


def _coerce_levels(raw: Any, name: str) -> TensorDict:
    if isinstance(raw, TensorDictBase):
        return (
            raw
            if all(torch.is_tensor(v) for v in raw.values())
            else _levels_td(_levels_list(raw))
        )
    if isinstance(raw, Mapping):
        return _levels_td(_levels_list(raw))
    if isinstance(raw, list | tuple):
        return _levels_td(list(raw))
    raise TypeError(f"{name} must be list, tuple, mapping, or TensorDict of tensors")


def _is_canonical_frame_features(feats: TensorDictBase) -> bool:
    if "vision_feats" not in feats or "vision_pos_embeds" not in feats:
        return False
    if not isinstance(feats["vision_feats"], TensorDictBase):
        return False
    if not isinstance(feats["vision_pos_embeds"], TensorDictBase):
        return False
    if feats.get_non_tensor("feat_sizes") is None:
        return False
    return True


def as_frame_features(feats: Mapping | TensorDictBase) -> TensorDict:
    """Canonical FrameFeatures TensorDict.

    Already-canonical TensorDict inputs are returned as-is (no rebuild).
    """
    if isinstance(feats, TensorDictBase) and _is_canonical_frame_features(feats):
        return feats

    if isinstance(feats, TensorDictBase):
        vision_feats_raw = feats["vision_feats"]
        vision_pos_raw = feats["vision_pos_embeds"]
        raw_feat_sizes = feats.get_non_tensor("feat_sizes")
        image = feats.get("image")
    elif isinstance(feats, Mapping):
        if "vision_feats" not in feats or "vision_pos_embeds" not in feats:
            raise ValueError(
                "vision_features entries need 'vision_feats' and 'vision_pos_embeds'"
            )
        vision_feats_raw = feats["vision_feats"]
        vision_pos_raw = feats["vision_pos_embeds"]
        raw_feat_sizes = feats.get("feat_sizes", None)
        image = feats.get("image", None)
    else:
        raise TypeError(f"unsupported frame feature type: {type(feats)!r}")

    vision_feats_td = _coerce_levels(vision_feats_raw, "vision_feats")
    vision_pos_td = _coerce_levels(vision_pos_raw, "vision_pos_embeds")
    if _sorted_numeric_keys(vision_feats_td) != _sorted_numeric_keys(vision_pos_td):
        raise ValueError(
            "vision_feats and vision_pos_embeds must have matching level keys"
        )

    levels = _levels_list(vision_feats_td)
    feat_sizes: list[tuple[int, int]]
    if raw_feat_sizes is None:
        if len(levels) == 0:
            raise ValueError("vision_feats is empty and feat_sizes missing")
        hw = levels[-1].shape[0]
        side = int(hw**0.5)
        if side * side != hw:
            raise ValueError(
                "cannot infer feat_sizes from non-square HW; pass feat_sizes"
            )
        feat_sizes = [(side, side)] * len(levels)
    elif isinstance(raw_feat_sizes, Sequence) and len(raw_feat_sizes) == len(levels):
        feat_sizes = []
        for size in raw_feat_sizes:
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise TypeError("feat_sizes entries must be pair-like (H, W)")
            feat_sizes.append((int(size[0]), int(size[1])))
    else:
        raise TypeError("feat_sizes must be a sequence of (H, W)")

    out = TensorDict(
        {
            "vision_feats": vision_feats_td,
            "vision_pos_embeds": vision_pos_td,
        },
        batch_size=[],
    )
    out = out.set_non_tensor("feat_sizes", feat_sizes)
    if image is not None:
        if not torch.is_tensor(image):
            raise TypeError(f"image must be a tensor, got {type(image)!r}")
        out["image"] = image
    return out


def unpack_frame_features(
    feats: Mapping | TensorDictBase,
    *,
    device: torch.device | str | None = None,
) -> tuple[list[Tensor], list[Tensor], list[tuple[int, int]], Tensor | None]:
    """Return unpacked frame features for ``track_step``.

    Returns ``(vision_feats, vision_pos_embeds, feat_sizes, image)``.
    """
    td = as_frame_features(feats)
    if device is not None:
        td = td.to(device=device, non_blocking=True)
    return (
        _levels_list(td["vision_feats"]),
        _levels_list(td["vision_pos_embeds"]),
        td.get_non_tensor("feat_sizes"),  # type: ignore[return-value]
        td.get("image"),
    )


def materialize_td(
    td: TensorDictBase | Mapping,
    device: torch.device | str,
) -> TensorDict:
    """Generic TensorDict materialization to a device."""
    if isinstance(td, TensorDictBase):
        source = td
    elif isinstance(td, Mapping):
        source = TensorDict(dict(td), batch_size=())
    else:
        raise TypeError(f"unexpected TensorDict/mapping type: {type(td)!r}")
    return source.to(device=device, non_blocking=True)


def memmap_td(td: TensorDictBase | Mapping, prefix: str | Path) -> TensorDict:
    """CPU memmap any TensorDict (features or outputs).

    Plain frame-feature dicts (list levels) are normalized via
    :func:`as_frame_features` first.
    """
    if isinstance(td, TensorDictBase):
        source = td
    elif isinstance(td, Mapping):
        # Frame-feature plain dicts carry list levels; normalize them.
        if "vision_feats" in td or "vision_pos_embeds" in td:
            source = as_frame_features(td)
        else:
            source = TensorDict(dict(td), batch_size=())
    else:
        raise TypeError(f"unexpected TensorDict/mapping type: {type(td)!r}")

    td_cpu = source.to("cpu", non_blocking=False)
    memmap_root = Path(prefix).expanduser()
    memmap_root.mkdir(parents=True, exist_ok=True)
    return td_cpu.memmap(prefix=memmap_root)


_FRAME_OUT_MEMMAP_KEYS = {
    "pred_masks",
    "pred_masks_high_res",
    "maskmem_features",
    "maskmem_pos_enc",
    "obj_ptr",
    "object_score_logits",
}


def memmap_frame_out(
    frame_out: dict[str, Any] | TensorDictBase, prefix: str | Path
) -> TensorDict:
    """Materialize/sparsify and memory-map selected heavy output tensors."""
    data: dict[str, Any] = {}
    if isinstance(frame_out, TensorDictBase):
        source = dict(frame_out.items())
    elif isinstance(frame_out, Mapping):
        source = dict(frame_out)
    else:
        raise TypeError(f"unexpected frame_out type: {type(frame_out)!r}")

    for key in _FRAME_OUT_MEMMAP_KEYS:
        if key not in source:
            continue
        value = source[key]
        if value is None:
            continue
        if key == "maskmem_pos_enc":
            if isinstance(value, TensorDictBase):
                data[key] = value
            elif isinstance(value, list | tuple):
                if len(value) == 0:
                    data[key] = TensorDict({}, batch_size=())
                elif all(isinstance(v, torch.Tensor) for v in value):
                    data[key] = TensorDict(
                        {str(i): t for i, t in enumerate(value)}, batch_size=()
                    )
                else:
                    continue
            else:
                continue
            continue
        if torch.is_tensor(value):
            data[key] = value
        elif isinstance(value, TensorDictBase):
            data[key] = value

    if not data:
        data = {key: value for key, value in source.items() if torch.is_tensor(value)}

    out = TensorDict(data, batch_size=())
    return memmap_td(out, prefix)
