"""Cut A — VisionTower I/O contracts (names, shapes, flat export order).

Single source of truth for export / runtime consumers. No model weights here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

VISION_IMAGE_SIZE: Final[int] = 1008
VISION_D_MODEL: Final[int] = 256
VISION_NUM_LEVELS: Final[int] = 4
# Spatial H=W per FPN level for production DualViTDet (S=1008, scales 4/2/1/0.5).
VISION_SPATIAL: Final[tuple[int, int, int, int]] = (288, 144, 72, 36)

FLAT_VISION_KEYS_SAM3: Final[tuple[str, ...]] = (
    "sam3_fpn_0",
    "sam3_fpn_1",
    "sam3_fpn_2",
    "sam3_fpn_3",
    "sam3_pe_0",
    "sam3_pe_1",
    "sam3_pe_2",
    "sam3_pe_3",
)
FLAT_VISION_KEYS_SAM2: Final[tuple[str, ...]] = (
    "sam2_fpn_0",
    "sam2_fpn_1",
    "sam2_fpn_2",
    "sam2_fpn_3",
    "sam2_pe_0",
    "sam2_pe_1",
    "sam2_pe_2",
    "sam2_pe_3",
)


@dataclass(frozen=True)
class VisionTowerSpec:
    """Metadata for validation and export wiring."""

    image_size: int = VISION_IMAGE_SIZE
    d_model: int = VISION_D_MODEL
    num_levels: int = VISION_NUM_LEVELS
    spatial: tuple[int, ...] = VISION_SPATIAL
    add_sam2: bool = True


@dataclass
class VisionTowerOutput:
    """Tensor-only multi-scale vision features (Cut A)."""

    sam3_fpn: tuple[Tensor, Tensor, Tensor, Tensor]
    sam3_pe: tuple[Tensor, Tensor, Tensor, Tensor]
    sam2_fpn: tuple[Tensor, Tensor, Tensor, Tensor] | None
    sam2_pe: tuple[Tensor, Tensor, Tensor, Tensor] | None


def flat_vision_keys(add_sam2: bool = True) -> tuple[str, ...]:
    if add_sam2:
        return FLAT_VISION_KEYS_SAM3 + FLAT_VISION_KEYS_SAM2
    return FLAT_VISION_KEYS_SAM3


def _ensure_level_tuple(
    block: tuple[Tensor, ...] | None,
    *,
    name: str,
    expected: int,
) -> tuple[Tensor, ...]:
    if block is None:
        raise ValueError(f"{name} must be a tuple of tensors, got None")
    if not isinstance(block, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(block)!r}")
    if len(block) != expected:
        raise ValueError(f"{name} must contain {expected} levels, got {len(block)}")
    return block


def _validate_level_tensor(
    tensor: Tensor,
    *,
    name: str,
    level: int,
    batch: int,
    d_model: int,
    hw: int,
    dtype: torch.dtype,
) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name}[{level}] must be Tensor, got {type(tensor)!r}")
    if tensor.ndim != 4:
        raise ValueError(
            f"{name}[{level}] must be rank-4 [B,C,H,W], got rank {tensor.ndim} "
            f"shape={tuple(tensor.shape)}"
        )
    b, c, h, w = tensor.shape
    if b != batch:
        raise ValueError(f"{name}[{level}] batch: expected {batch}, got {b}")
    if c != d_model:
        raise ValueError(f"{name}[{level}] channels: expected {d_model}, got {c}")
    if h != hw or w != hw:
        raise ValueError(f"{name}[{level}] spatial: expected {(hw, hw)}, got {(h, w)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name}[{level}] dtype: expected {dtype}, got {tensor.dtype}")


def vision_output_to_flat(out: VisionTowerOutput) -> tuple[Tensor, ...]:
    """Flatten in ``flat_vision_keys`` order."""
    if not isinstance(out, VisionTowerOutput):
        raise TypeError(f"out must be VisionTowerOutput, got {type(out)!r}")

    sam3_fpn = _ensure_level_tuple(
        out.sam3_fpn, name="sam3_fpn", expected=VISION_NUM_LEVELS
    )
    sam3_pe = _ensure_level_tuple(
        out.sam3_pe, name="sam3_pe", expected=VISION_NUM_LEVELS
    )

    if (out.sam2_fpn is None) ^ (out.sam2_pe is None):
        raise ValueError("sam2_fpn and sam2_pe must both be set or both be None")

    if out.sam2_fpn is None:
        return sam3_fpn + sam3_pe

    sam2_fpn = _ensure_level_tuple(
        out.sam2_fpn, name="sam2_fpn", expected=VISION_NUM_LEVELS
    )
    sam2_pe = _ensure_level_tuple(
        out.sam2_pe, name="sam2_pe", expected=VISION_NUM_LEVELS
    )
    return sam3_fpn + sam3_pe + sam2_fpn + sam2_pe


def vision_output_from_flat(
    tensors: tuple[Tensor, ...] | list[Tensor],
    *,
    add_sam2: bool = True,
) -> VisionTowerOutput:
    """Inverse of ``vision_output_to_flat``; validates shapes."""
    if not isinstance(tensors, (tuple, list)):
        raise TypeError(f"tensors must be a sequence, got {type(tensors)!r}")
    tensors_t = tuple(tensors)
    expected = 2 * VISION_NUM_LEVELS * (2 if add_sam2 else 1)
    if len(tensors_t) != expected:
        raise ValueError(
            f"expected {expected} tensors for add_sam2={add_sam2}, got {len(tensors_t)}"
        )
    for i, t in enumerate(tensors_t):
        if not torch.is_tensor(t):
            raise TypeError(f"tensors[{i}] must be Tensor, got {type(t)!r}")

    n = VISION_NUM_LEVELS
    sam3_fpn = tensors_t[0:n]
    sam3_pe = tensors_t[n : 2 * n]
    if add_sam2:
        sam2_fpn = tensors_t[2 * n : 3 * n]
        sam2_pe = tensors_t[3 * n : 4 * n]
    else:
        sam2_fpn = None
        sam2_pe = None

    out = VisionTowerOutput(
        sam3_fpn=sam3_fpn,  # type: ignore[arg-type]
        sam3_pe=sam3_pe,  # type: ignore[arg-type]
        sam2_fpn=sam2_fpn,  # type: ignore[arg-type]
        sam2_pe=sam2_pe,  # type: ignore[arg-type]
    )
    validate_vision_output(
        out,
        batch=int(tensors_t[0].shape[0]),
        dtype=tensors_t[0].dtype,
        spec=VisionTowerSpec(add_sam2=add_sam2),
    )
    return out


def validate_vision_input(
    pixel_values: Tensor,
    spec: VisionTowerSpec = VisionTowerSpec(),
) -> None:
    if not torch.is_tensor(pixel_values):
        raise TypeError(f"pixel_values must be Tensor, got {type(pixel_values)!r}")
    if pixel_values.ndim != 4:
        raise ValueError(
            f"pixel_values must be [B,3,H,W], got rank {pixel_values.ndim} "
            f"shape={tuple(pixel_values.shape)}"
        )
    if pixel_values.shape[1] != 3:
        raise ValueError(f"pixel_values C must be 3, got {pixel_values.shape[1]}")
    h, w = int(pixel_values.shape[2]), int(pixel_values.shape[3])
    if h != spec.image_size or w != spec.image_size:
        raise ValueError(
            f"pixel_values spatial must be {(spec.image_size, spec.image_size)}, "
            f"got {(h, w)}"
        )
    if not pixel_values.is_floating_point():
        raise ValueError(
            f"pixel_values must be floating dtype, got {pixel_values.dtype}"
        )


def validate_vision_output(
    out: VisionTowerOutput,
    *,
    batch: int,
    dtype: torch.dtype,
    spec: VisionTowerSpec = VisionTowerSpec(),
) -> None:
    if not isinstance(out, VisionTowerOutput):
        raise TypeError(f"out must be VisionTowerOutput, got {type(out)!r}")
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError(f"batch must be positive int, got {batch!r}")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be torch.dtype, got {type(dtype)!r}")
    if len(spec.spatial) != spec.num_levels:
        raise ValueError(
            f"spec.spatial length {len(spec.spatial)} != num_levels {spec.num_levels}"
        )

    sam3_fpn = _ensure_level_tuple(
        out.sam3_fpn, name="sam3_fpn", expected=spec.num_levels
    )
    sam3_pe = _ensure_level_tuple(out.sam3_pe, name="sam3_pe", expected=spec.num_levels)

    blocks: list[tuple[str, tuple[Tensor, ...]]] = [
        ("sam3_fpn", sam3_fpn),
        ("sam3_pe", sam3_pe),
    ]

    if spec.add_sam2:
        sam2_fpn = _ensure_level_tuple(
            out.sam2_fpn, name="sam2_fpn", expected=spec.num_levels
        )
        sam2_pe = _ensure_level_tuple(
            out.sam2_pe, name="sam2_pe", expected=spec.num_levels
        )
        blocks.extend([("sam2_fpn", sam2_fpn), ("sam2_pe", sam2_pe)])
    else:
        if out.sam2_fpn is not None or out.sam2_pe is not None:
            raise ValueError("sam2_* must be None when spec.add_sam2 is False")

    for name, block in blocks:
        for level, hw in enumerate(spec.spatial):
            _validate_level_tensor(
                block[level],
                name=name,
                level=level,
                batch=batch,
                d_model=spec.d_model,
                hw=int(hw),
                dtype=dtype,
            )


__all__ = [
    "VISION_IMAGE_SIZE",
    "VISION_D_MODEL",
    "VISION_NUM_LEVELS",
    "VISION_SPATIAL",
    "FLAT_VISION_KEYS_SAM3",
    "FLAT_VISION_KEYS_SAM2",
    "VisionTowerSpec",
    "VisionTowerOutput",
    "flat_vision_keys",
    "vision_output_to_flat",
    "vision_output_from_flat",
    "validate_vision_input",
    "validate_vision_output",
]
