"""Tensor-only object-space/bucket-space components for SAM3.1 Multiplex."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sam3.runtime.multiplex_state import BUCKET_CAPACITY


def _flat_assignment(
    assignments: Tensor, bucket_count: int, *, validate_values: bool = True
) -> Tensor:
    if assignments.dtype != torch.int64:
        raise TypeError("assignments must use int64")
    if assignments.ndim != 2 or assignments.shape[1] != 2:
        raise ValueError("assignments must have shape [objects,2]")
    buckets = assignments[:, 0]
    slots = assignments[:, 1]
    flat = buckets * BUCKET_CAPACITY + slots
    if validate_values:
        torch._assert(torch.all(buckets >= 0), "bucket index must be non-negative")
        torch._assert(torch.all(buckets < bucket_count), "bucket index is out of range")
        torch._assert(torch.all(slots >= 0), "slot index must be non-negative")
        torch._assert(torch.all(slots < BUCKET_CAPACITY), "slot index is out of range")
        torch._assert(
            torch.unique(flat).numel() == flat.numel(),
            "assignments must identify unique bucket slots",
        )
    return flat


class Mux(nn.Module):
    """Scatter object-space values into fixed bucket/slot space."""

    def __init__(self, bucket_count: int) -> None:
        super().__init__()
        if bucket_count not in (1, 2):
            raise ValueError("bucket_count must be 1 or 2")
        self.bucket_count = bucket_count

    def forward(
        self, object_values: Tensor, assignments: Tensor, slot_validity: Tensor
    ) -> Tensor:
        flat = _flat_assignment(assignments, self.bucket_count)
        if object_values.shape[0] != assignments.shape[0]:
            raise ValueError("object_values and assignments disagree on object count")
        if slot_validity.shape != (self.bucket_count, BUCKET_CAPACITY):
            raise ValueError("slot_validity has the wrong fixed bucket shape")
        output = object_values.new_zeros(
            (self.bucket_count * BUCKET_CAPACITY, *object_values.shape[1:])
        )
        output = output.index_copy(0, flat, object_values)
        output = output.view(
            self.bucket_count, BUCKET_CAPACITY, *object_values.shape[1:]
        )
        validity = slot_validity.to(torch.bool)
        validity = validity.view(
            self.bucket_count,
            BUCKET_CAPACITY,
            *((1,) * (object_values.ndim - 1)),
        )
        return torch.where(validity, output, torch.zeros_like(output))


class Demux(nn.Module):
    """Gather fixed bucket/slot values into host-selected object order."""

    def __init__(self, bucket_count: int) -> None:
        super().__init__()
        if bucket_count not in (1, 2):
            raise ValueError("bucket_count must be 1 or 2")
        self.bucket_count = bucket_count

    def forward(
        self, bucket_values: Tensor, assignments: Tensor, slot_validity: Tensor
    ) -> Tensor:
        flat = _flat_assignment(assignments, self.bucket_count)
        if bucket_values.shape[:2] != (self.bucket_count, BUCKET_CAPACITY):
            raise ValueError("bucket_values has the wrong fixed bucket shape")
        if slot_validity.shape != (self.bucket_count, BUCKET_CAPACITY):
            raise ValueError("slot_validity has the wrong fixed bucket shape")
        flattened = bucket_values.flatten(0, 1)
        selected = flattened.index_select(0, flat)
        selected_validity = slot_validity.to(torch.bool).flatten().index_select(0, flat)
        selected_validity = selected_validity.view(
            selected.shape[0], *((1,) * (selected.ndim - 1))
        )
        return torch.where(selected_validity, selected, torch.zeros_like(selected))


class ScatterReplace(nn.Module):
    """Replace one selected slot while preserving every non-target byte."""

    def __init__(
        self, bucket_count: int | None, *, validate_assignments: bool = True
    ) -> None:
        super().__init__()
        if bucket_count not in (1, 2, None):
            raise ValueError("bucket_count must be 1, 2, or bounded-dynamic")
        if bucket_count is None and validate_assignments:
            raise ValueError("dynamic private scatter requires host validation")
        self.bucket_count = bucket_count
        self.validate_assignments = bool(validate_assignments)

    def forward(
        self, bucket_values: Tensor, replacement: Tensor, assignment: Tensor
    ) -> Tensor:
        bucket_count = (
            bucket_values.shape[0]
            if self.bucket_count is None
            else self.bucket_count
        )
        flat = _flat_assignment(
            assignment,
            bucket_count,
            validate_values=self.validate_assignments,
        )
        if assignment.shape[0] != 1 or replacement.shape[0] != 1:
            raise ValueError("scatter replace accepts exactly one selected object")
        if (
            self.bucket_count is not None
            and bucket_values.shape[:2] != (self.bucket_count, BUCKET_CAPACITY)
        ):
            raise ValueError("bucket_values has the wrong fixed bucket shape")
        output = bucket_values.flatten(0, 1)
        output = output.index_copy(0, flat, replacement)
        return output.view_as(bucket_values)


__all__ = ["Demux", "Mux", "ScatterReplace"]
