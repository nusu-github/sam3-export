"""Small helpers used by the SAM3 video tracker.

Eval-critical utilities used by the video track path:
- ``select_closest_cond_frames`` — temporal locality for cond-frame attention
- ``get_1d_sine_pe`` — 1-D sine temporal positional encoding for object pointers
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jaxtyping import Float
import torch
from torch import Tensor


def select_closest_cond_frames(
    frame_idx: int,
    cond_frame_outputs: Mapping[int, Any],
    max_cond_frame_num: int,
    keep_first_cond_frame: bool = False,
) -> tuple[Mapping[int, Any], dict[int, Any]]:
    """
    Select up to ``max_cond_frame_num`` conditioning frames from ``cond_frame_outputs``
    that are temporally closest to the current frame at ``frame_idx``. Here, we take
    - a) the closest conditioning frame before ``frame_idx`` (if any);
    - b) the closest conditioning frame after ``frame_idx`` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of ``max_cond_frame_num`` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from ``cond_frame_outputs``.
    - unselected_outputs: items (keys & values) not selected in ``cond_frame_outputs``.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs: Mapping[int, Any] = cond_frame_outputs
        unselected_outputs: dict[int, Any] = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        if keep_first_cond_frame:
            idx_first = min(
                (t for t in cond_frame_outputs if t < frame_idx), default=None
            )
            if idx_first is None:
                # Maybe we are tracking in reverse
                idx_first = max(
                    (t for t in cond_frame_outputs if t > frame_idx), default=None
                )
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        # the closest conditioning frame before ``frame_idx`` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after ``frame_idx`` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of ``max_cond_frame_num`` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(
    pos_inds: Float[Tensor, "*batch"],
    dim: int,
    temperature: float | int = 10000,
) -> Float[Tensor, "*batch pe_dim"]:
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed
