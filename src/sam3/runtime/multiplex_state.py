"""Host-owned assignment state for the SAM3.1 Multiplex plan.

Only object identity and lifecycle live here.  Learned bucket tensors are owned by
the backend adapter and deliberately have no representation in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

MULTIPLEX_STATE_ABI: Final[str] = "MultiplexStateV1"
BUCKET_CAPACITY: Final[int] = 16
MAX_BUCKETS: Final[int] = 2
MAX_OBJECTS: Final[int] = BUCKET_CAPACITY * MAX_BUCKETS
CONDITIONING_CAPACITY: Final[int] = 4
NON_CONDITIONING_CAPACITY: Final[int] = 6
SPATIAL_CAPACITY: Final[int] = CONDITIONING_CAPACITY + NON_CONDITIONING_CAPACITY
POINTER_FRAME_CAPACITY: Final[int] = 16


@dataclass(frozen=True)
class MultiplexVariantParameters:
    """M5 graph values owned by the checkpoint or fixed official builder."""

    bucket_capacity: int
    max_buckets: int
    num_maskmem: int
    conditioning_spatial_capacity: int
    non_conditioning_spatial_capacity: int
    total_spatial_input_capacity: int
    object_pointer_frame_capacity: int
    hidden_dimension: int
    memory_dimension: int
    memory_spatial_size: tuple[int, int]
    image_size: int
    mask_candidates: int
    memory_mask_channels: int
    memory_sigmoid_scale: float
    memory_sigmoid_bias: float
    condition_mask_foreground: float
    condition_mask_background: float
    non_overlap_memory: bool

    def validate(self) -> None:
        expected: dict[str, object] = {
            "bucket_capacity": BUCKET_CAPACITY,
            "max_buckets": MAX_BUCKETS,
            "num_maskmem": 7,
            "conditioning_spatial_capacity": CONDITIONING_CAPACITY,
            "non_conditioning_spatial_capacity": NON_CONDITIONING_CAPACITY,
            "total_spatial_input_capacity": SPATIAL_CAPACITY,
            "object_pointer_frame_capacity": POINTER_FRAME_CAPACITY,
            "hidden_dimension": 256,
            "memory_dimension": 256,
            "memory_spatial_size": (72, 72),
            "image_size": 1008,
            "mask_candidates": 3,
            "memory_mask_channels": 32,
            "memory_sigmoid_scale": 2.0,
            "memory_sigmoid_bias": -1.0,
            "condition_mask_foreground": 1.0,
            "condition_mask_background": 0.0,
            "non_overlap_memory": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"unsupported MultiplexStateV1 variant parameter {name}: "
                    f"{getattr(self, name)!r} != {value!r}"
                )

    @classmethod
    def native(cls) -> MultiplexVariantParameters:
        result = cls(
            bucket_capacity=BUCKET_CAPACITY,
            max_buckets=MAX_BUCKETS,
            num_maskmem=7,
            conditioning_spatial_capacity=CONDITIONING_CAPACITY,
            non_conditioning_spatial_capacity=NON_CONDITIONING_CAPACITY,
            total_spatial_input_capacity=SPATIAL_CAPACITY,
            object_pointer_frame_capacity=POINTER_FRAME_CAPACITY,
            hidden_dimension=256,
            memory_dimension=256,
            memory_spatial_size=(72, 72),
            image_size=1008,
            mask_candidates=3,
            memory_mask_channels=32,
            memory_sigmoid_scale=2.0,
            memory_sigmoid_bias=-1.0,
            condition_mask_foreground=1.0,
            condition_mask_background=0.0,
            non_overlap_memory=False,
        )
        result.validate()
        return result

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> MultiplexVariantParameters:
        values = {
            str(item["name"]): item["value"]
            for item in manifest["model"]["variant_parameters"]
        }
        required = {
            "bucket-capacity": "bucket_capacity",
            "max-buckets": "max_buckets",
            "num-maskmem": "num_maskmem",
            "conditioning-spatial-capacity": "conditioning_spatial_capacity",
            "non-conditioning-spatial-capacity": ("non_conditioning_spatial_capacity"),
            "total-spatial-input-capacity": "total_spatial_input_capacity",
            "object-pointer-frame-capacity": "object_pointer_frame_capacity",
            "hidden-dimension": "hidden_dimension",
            "memory-dimension": "memory_dimension",
            "memory-spatial-size": "memory_spatial_size",
            "image-size": "image_size",
            "mask-candidates": "mask_candidates",
            "memory-mask-channels": "memory_mask_channels",
            "memory-sigmoid-scale": "memory_sigmoid_scale",
            "memory-sigmoid-bias": "memory_sigmoid_bias",
            "condition-mask-foreground": "condition_mask_foreground",
            "condition-mask-background": "condition_mask_background",
            "non-overlap-memory": "non_overlap_memory",
        }
        missing = sorted(set(required) - set(values))
        if missing:
            raise ValueError(f"M5 manifest is missing variant parameters: {missing}")
        kwargs = {attribute: values[name] for name, attribute in required.items()}
        raw_size = kwargs["memory_spatial_size"]
        if not isinstance(raw_size, list) or len(raw_size) != 2:
            raise ValueError("memory-spatial-size must be [height, width]")
        kwargs["memory_spatial_size"] = (int(raw_size[0]), int(raw_size[1]))
        result = cls(**kwargs)
        result.validate()
        return result


class MultiplexCapacityError(RuntimeError):
    """The public two-bucket capacity would be exceeded."""


@dataclass(frozen=True)
class SlotAssignment:
    """Private host-runtime location for one public object ID."""

    bucket: int
    slot: int


class MultiplexStateV1:
    """Stable object-to-slot assignment and assignment revision discipline."""

    def __init__(self) -> None:
        self._assignments: dict[int, SlotAssignment] = {}
        self._slots: list[list[int | None]] = [
            [None] * BUCKET_CAPACITY for _ in range(MAX_BUCKETS)
        ]
        self.revision = 0

    @property
    def object_count(self) -> int:
        return len(self._assignments)

    @property
    def bucket_count(self) -> int:
        if not self._assignments:
            return 0
        return max(value.bucket for value in self._assignments.values()) + 1

    @property
    def object_ids(self) -> tuple[int, ...]:
        """Return public IDs in the deterministic final-result order."""

        return tuple(sorted(self._assignments))

    def _validate_new_object_id(self, object_id: int) -> None:
        if not isinstance(object_id, int) or isinstance(object_id, bool):
            raise TypeError("object_id must be an integer")
        if object_id in self._assignments:
            raise ValueError(f"duplicate object ID: {object_id}")

    def add_object(self, object_id: int) -> SlotAssignment:
        self._validate_new_object_id(object_id)
        for bucket, row in enumerate(self._slots):
            for slot, assigned in enumerate(row):
                if assigned is None:
                    location = SlotAssignment(bucket=bucket, slot=slot)
                    row[slot] = object_id
                    self._assignments[object_id] = location
                    self.revision += 1
                    return location
        raise MultiplexCapacityError(
            f"MultiplexStateV1 capacity is {MAX_OBJECTS} objects"
        )

    def require_object(self, object_id: int) -> SlotAssignment:
        try:
            return self._assignments[object_id]
        except KeyError as exc:
            raise KeyError(f"unknown object ID: {object_id}") from exc

    def remove_object(self, object_id: int) -> SlotAssignment:
        location = self.require_object(object_id)
        del self._assignments[object_id]
        self._slots[location.bucket][location.slot] = None
        self.revision += 1
        return location

    def replace_object(self, object_id: int, replacement_id: int) -> SlotAssignment:
        """Replace identity in-place without compacting either bucket."""

        location = self.require_object(object_id)
        self._validate_new_object_id(replacement_id)
        del self._assignments[object_id]
        self._assignments[replacement_id] = location
        self._slots[location.bucket][location.slot] = replacement_id
        self.revision += 1
        return location

    def assignment_array(
        self, object_ids: tuple[int, ...] | list[int] | None = None
    ) -> np.ndarray:
        """Return private tensor-component input in the requested object order."""

        selected = self.object_ids if object_ids is None else tuple(object_ids)
        locations = [self.require_object(object_id) for object_id in selected]
        return np.asarray(
            [(value.bucket, value.slot) for value in locations], dtype=np.int64
        ).reshape((-1, 2))

    def validity_array(self, *, bucket_count: int | None = None) -> np.ndarray:
        count = self.bucket_count if bucket_count is None else bucket_count
        if count not in (1, 2):
            raise ValueError("bucket_count must be 1 or 2")
        result = np.zeros((count, BUCKET_CAPACITY), dtype=np.bool_)
        for location in self._assignments.values():
            if location.bucket < count:
                result[location.bucket, location.slot] = True
        return result


__all__ = [
    "BUCKET_CAPACITY",
    "CONDITIONING_CAPACITY",
    "MAX_BUCKETS",
    "MAX_OBJECTS",
    "MULTIPLEX_STATE_ABI",
    "MultiplexVariantParameters",
    "MultiplexCapacityError",
    "MultiplexStateV1",
    "NON_CONDITIONING_CAPACITY",
    "POINTER_FRAME_CAPACITY",
    "SPATIAL_CAPACITY",
    "SlotAssignment",
]
