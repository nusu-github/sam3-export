"""Spatial and position helpers for ViTDet.

Generic pieces come from ``timm`` (same source official SAM3 uses for
``DropPath`` / ``trunc_normal_``). SAM3-only absolute-pos tiling and
concat-rel-pos stay local.
"""

from __future__ import annotations

import math
from typing import Optional

from jaxtyping import Float, Integer

# Official SAM3 already depends on these; no hand-rolled copies.
from timm.layers import DropPath, LayerScale, trunc_normal_
import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "DropPath",
    "LayerScale",
    "trunc_normal_",
    "window_partition",
    "window_unpartition",
    "get_rel_pos",
    "get_abs_pos",
    "concat_rel_pos",
]


def window_partition(
    x: Float[Tensor, "b h w c"], window_size: int
) -> tuple[Float[Tensor, "num_windows win win c"], tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Args:
        x: Input token grid ``(B, H, W, C)``.
        window_size: Window height/width.

    Returns:
        windows: ``(B * num_windows, window_size, window_size, C)``.
        (Hp, Wp): Padded spatial dimensions before partition.
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: Float[Tensor, "num_windows win win c"],
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> Float[Tensor, "b h w c"]:
    """Undo ``window_partition`` and remove padding.

    Detectron2/SAM3 argument order: ``pad_hw`` then ``hw`` (timm's SAM ViT
    uses the reverse order).
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(
    q_size: int, k_size: int, rel_pos: Float[Tensor, "l d"]
) -> Float[Tensor, "q_size k_size d"]:
    """Relative positional embeddings for query/key spans (ViTDet / SAM)."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
            align_corners=False,
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device, dtype=torch.float32)[
        :, None
    ] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device, dtype=torch.float32)[
        None, :
    ] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def get_abs_pos(
    abs_pos: Float[Tensor, "b n c"],
    has_cls_token: bool,
    hw: tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> Float[Tensor, "1 h w c"] | Float[Tensor, "1 seq c"]:
    """Resize absolute positional embeddings; optional SAM3 tiling path."""
    if retain_cls_token:
        assert has_cls_token

    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile(
                [1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])]
            )[:, :, :h, :w]
        else:
            new_abs_pos = F.interpolate(
                new_abs_pos,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

        if not retain_cls_token:
            return new_abs_pos.permute(0, 2, 3, 1)
        assert has_cls_token
        return torch.cat(
            [cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)],
            dim=1,
        )

    if not retain_cls_token:
        return abs_pos.reshape(1, h, w, -1)
    assert has_cls_token
    return torch.cat([cls_pos, abs_pos], dim=1)


def concat_rel_pos(
    q: Float[Tensor, "b n d"],
    k: Float[Tensor, "b nk d"],
    q_hw: tuple[int, int],
    k_hw: tuple[int, int],
    rel_pos_h: Float[Tensor, "lh d"],
    rel_pos_w: Float[Tensor, "lw d"],
    rescale: bool = False,
    relative_coords: Optional[Integer[Tensor, "..."]] = None,
) -> tuple[Float[Tensor, "b n d_out"], Float[Tensor, "b nk d_out"]]:
    """Concatenate rel-pos tensors so the QK product carries positional bias."""
    q_h, q_w = q_hw
    k_h, k_w = k_hw
    assert (q_h == q_w) and (k_h == k_w), "only square inputs supported"

    if relative_coords is not None:
        Rh = rel_pos_h[relative_coords]
        Rw = rel_pos_w[relative_coords]
    else:
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale
    scale_ratio = new_scale / old_scale

    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) * new_scale
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) * new_scale

    eye_h = (
        torch.eye(k_h, dtype=q.dtype, device=q.device)
        .view(1, k_h, 1, k_h)
        .expand([B, k_h, k_w, k_h])
    )
    eye_w = (
        torch.eye(k_w, dtype=q.dtype, device=q.device)
        .view(1, 1, k_w, k_w)
        .expand([B, k_h, k_w, k_w])
    )

    q = torch.cat([r_q * scale_ratio, rel_h, rel_w], dim=-1).view(B, q_h * q_w, -1)
    k = torch.cat([k.view(B, k_h, k_w, -1), eye_h, eye_w], dim=-1).view(
        B, k_h * k_w, -1
    )
    return q, k
