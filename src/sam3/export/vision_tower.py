"""Cut A — VisionTower wrappers over ``Sam3DualViTDetNeck``."""

from __future__ import annotations

from torch import Tensor
import torch.nn as nn

from sam3.export.contracts import (
    VisionTowerOutput,
    VisionTowerSpec,
    validate_vision_input,
    validate_vision_output,
    vision_output_to_flat,
)
from sam3.vision.necks import Sam3DualViTDetNeck


def _as_level_tuple(values: list[Tensor] | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"expected list/tuple of tensors, got {type(values)!r}")
    return tuple(values)


class VisionTower(nn.Module):
    """Named multi-scale vision features (eager / runtime)."""

    def __init__(
        self,
        neck: Sam3DualViTDetNeck,
        spec: VisionTowerSpec | None = None,
        *,
        validate: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(neck, Sam3DualViTDetNeck):
            raise TypeError(f"neck must be Sam3DualViTDetNeck, got {type(neck)!r}")
        if spec is None:
            spec = VisionTowerSpec(add_sam2=neck.sam2_convs is not None)
        self.neck = neck
        self.spec = spec
        self.validate = bool(validate)

    def forward(
        self,
        pixel_values: Tensor,
        *,
        validate: bool | None = None,
    ) -> VisionTowerOutput:
        do_validate = self.validate if validate is None else bool(validate)
        if do_validate:
            validate_vision_input(pixel_values, spec=self.spec)

        sam3_fpn, sam3_pe, sam2_fpn, sam2_pe = self.neck(pixel_values)
        out = VisionTowerOutput(
            sam3_fpn=_as_level_tuple(sam3_fpn),  # type: ignore[arg-type]
            sam3_pe=_as_level_tuple(sam3_pe),  # type: ignore[arg-type]
            sam2_fpn=None if sam2_fpn is None else _as_level_tuple(sam2_fpn),  # type: ignore[arg-type]
            sam2_pe=None if sam2_pe is None else _as_level_tuple(sam2_pe),  # type: ignore[arg-type]
        )
        if do_validate:
            validate_vision_output(
                out,
                batch=int(pixel_values.shape[0]),
                dtype=pixel_values.dtype,
                spec=self.spec,
            )
        return out

    def forward_flat(
        self,
        pixel_values: Tensor,
        *,
        validate: bool | None = None,
    ) -> tuple[Tensor, ...]:
        return vision_output_to_flat(self.forward(pixel_values, validate=validate))


class VisionTowerFlat(nn.Module):
    """Export-oriented entry: ``forward`` returns a flat tensor tuple only."""

    def __init__(
        self,
        neck: Sam3DualViTDetNeck,
        spec: VisionTowerSpec | None = None,
    ) -> None:
        super().__init__()
        # Skip nested validation inside export tracing.
        self._tower = VisionTower(neck, spec=spec, validate=False)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        return self._tower.forward_flat(pixel_values, validate=False)


__all__ = ["VisionTower", "VisionTowerFlat"]
