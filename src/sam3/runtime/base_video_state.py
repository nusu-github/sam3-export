"""Host-owned fixed-capacity state for the M4 SAM3 base video plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

STATE_ABI_VERSION: Final[str] = "BaseVideoStateV1"
CONDITIONING_CAPACITY: Final[int] = 4
NON_CONDITIONING_CAPACITY: Final[int] = 6
SPATIAL_CAPACITY: Final[int] = 10
POINTER_CAPACITY: Final[int] = 16


class StateCapacityError(RuntimeError):
    """A fixed BaseVideoStateV1 capacity would be exceeded."""


@dataclass(frozen=True)
class BaseVideoVariantParameters:
    """Checkpoint-owned values required by every M4 production graph."""

    num_maskmem: int
    conditioning_spatial_capacity: int
    non_conditioning_spatial_capacity: int
    total_spatial_input_capacity: int
    object_pointer_capacity: int
    hidden_dimension: int
    memory_dimension: int
    memory_spatial_size: tuple[int, int]
    temporal_stride: int
    memory_sigmoid_scale: float
    memory_sigmoid_bias: float
    non_overlap_memory: bool

    def validate(self) -> None:
        expected: dict[str, object] = {
            "num_maskmem": 7,
            "conditioning_spatial_capacity": CONDITIONING_CAPACITY,
            "non_conditioning_spatial_capacity": NON_CONDITIONING_CAPACITY,
            "total_spatial_input_capacity": SPATIAL_CAPACITY,
            "object_pointer_capacity": POINTER_CAPACITY,
            "hidden_dimension": 256,
            "memory_dimension": 64,
            "memory_spatial_size": (72, 72),
            "temporal_stride": 1,
            "memory_sigmoid_scale": 20.0,
            "memory_sigmoid_bias": -10.0,
            "non_overlap_memory": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"unsupported BaseVideoStateV1 variant parameter {name}: "
                    f"{getattr(self, name)!r} != {value!r}"
                )

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> BaseVideoVariantParameters:
        values = {
            str(item["name"]): item["value"]
            for item in manifest["model"]["variant_parameters"]
        }
        required = {
            "num-maskmem": "num_maskmem",
            "conditioning-spatial-capacity": "conditioning_spatial_capacity",
            "non-conditioning-spatial-capacity": ("non_conditioning_spatial_capacity"),
            "total-spatial-input-capacity": "total_spatial_input_capacity",
            "object-pointer-capacity": "object_pointer_capacity",
            "hidden-dimension": "hidden_dimension",
            "memory-dimension": "memory_dimension",
            "memory-spatial-size": "memory_spatial_size",
            "temporal-stride": "temporal_stride",
            "memory-sigmoid-scale": "memory_sigmoid_scale",
            "memory-sigmoid-bias": "memory_sigmoid_bias",
            "non-overlap-memory": "non_overlap_memory",
        }
        missing = sorted(set(required) - set(values))
        if missing:
            raise ValueError(f"M4 manifest is missing variant parameters: {missing}")
        kwargs = {attribute: values[name] for name, attribute in required.items()}
        raw_size = kwargs["memory_spatial_size"]
        if not isinstance(raw_size, list) or len(raw_size) != 2:
            raise ValueError("memory-spatial-size must be [height, width]")
        kwargs["memory_spatial_size"] = (int(raw_size[0]), int(raw_size[1]))
        result = cls(**kwargs)
        result.validate()
        return result


@dataclass
class VideoStateEntry:
    """One committed per-object frame entry; tensor values remain backend-owned."""

    frame_index: int
    conditioning: bool
    memory_features: object
    memory_position: object
    object_pointer: object
    mask: np.ndarray
    score: float
    low_res_logits: np.ndarray
    object_score: float


@dataclass
class ObjectVideoState:
    object_id: int
    conditioning: dict[int, VideoStateEntry] = field(default_factory=dict)
    non_conditioning: dict[int, VideoStateEntry] = field(default_factory=dict)
    last_low_res_logits: np.ndarray | None = None
    direction: int = 1


@dataclass(frozen=True)
class PackedObjectState:
    """One fixed-capacity chunk passed to the backend adapter."""

    object_ids: tuple[int | None, ...]
    object_valid: np.ndarray
    spatial_entries: tuple[tuple[VideoStateEntry | None, ...], ...]
    memory_valid: np.ndarray
    memory_age: np.ndarray
    memory_conditioning: np.ndarray
    pointer_entries: tuple[tuple[VideoStateEntry | None, ...], ...]
    pointer_valid: np.ndarray
    pointer_age: np.ndarray
    pointer_conditioning: np.ndarray
    pointer_tpos_denominator: np.ndarray


def _closest_conditioning(
    entries: dict[int, VideoStateEntry], frame_index: int
) -> tuple[list[VideoStateEntry], dict[int, VideoStateEntry]]:
    if len(entries) <= CONDITIONING_CAPACITY:
        return list(entries.values()), {}
    selected_frames: set[int] = set()
    before = [value for value in entries if value < frame_index]
    after = [value for value in entries if value >= frame_index]
    if before:
        selected_frames.add(max(before))
    if after:
        selected_frames.add(min(after))
    remaining = sorted(
        (value for value in entries if value not in selected_frames),
        key=lambda value: abs(value - frame_index),
    )
    selected_frames.update(remaining[: CONDITIONING_CAPACITY - len(selected_frames)])
    selected = [entries[value] for value in entries if value in selected_frames]
    unselected = {
        value: entry for value, entry in entries.items() if value not in selected_frames
    }
    return selected, unselected


class BaseVideoStateV1:
    """Per-object state selection and revision rules for the M4 base plan."""

    def __init__(self, *, batch_capacity: int) -> None:
        if batch_capacity not in (4, 8):
            raise ValueError("BaseVideoStateV1 batch capacity must be B4 or B8")
        self.batch_capacity = batch_capacity
        self.objects: dict[int, ObjectVideoState] = {}
        self.revision = 0

    def add_object(self, object_id: int) -> None:
        if not isinstance(object_id, int) or isinstance(object_id, bool):
            raise TypeError("object_id must be an integer")
        if object_id in self.objects:
            raise ValueError(f"duplicate object ID: {object_id}")
        self.objects[object_id] = ObjectVideoState(object_id=object_id)

    def require_object(self, object_id: int) -> ObjectVideoState:
        try:
            return self.objects[object_id]
        except KeyError as exc:
            raise KeyError(f"unknown object ID: {object_id}") from exc

    def commit(self, object_id: int, entry: VideoStateEntry) -> None:
        state = self.require_object(object_id)
        if entry.conditioning:
            replacing = entry.frame_index in state.conditioning
            if not replacing and len(state.conditioning) >= CONDITIONING_CAPACITY:
                raise StateCapacityError(
                    f"object {object_id} conditioning capacity is "
                    f"{CONDITIONING_CAPACITY}"
                )
            state.conditioning[entry.frame_index] = entry
            state.non_conditioning.pop(entry.frame_index, None)
            if state.direction > 0:
                stale = [
                    value
                    for value in state.non_conditioning
                    if value > entry.frame_index
                ]
            else:
                stale = [
                    value
                    for value in state.non_conditioning
                    if value < entry.frame_index
                ]
            for value in stale:
                del state.non_conditioning[value]
        else:
            state.non_conditioning[entry.frame_index] = entry
            state.direction = 1 if state.direction >= 0 else -1
        state.last_low_res_logits = entry.low_res_logits
        self.revision += 1

    def pack(
        self,
        object_ids: list[int],
        *,
        frame_index: int,
        reverse: bool,
        video_frame_count: int | None = None,
    ) -> list[PackedObjectState]:
        if not object_ids:
            return []
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("object_ids must be unique")
        for object_id in object_ids:
            self.require_object(object_id)
        direction = -1 if reverse else 1
        if video_frame_count is None:
            video_frame_count = max(frame_index + 1, 1)
        if video_frame_count <= 0:
            raise ValueError("video_frame_count must be positive")
        pointer_tpos_denominator = max(min(video_frame_count, POINTER_CAPACITY) - 1, 1)
        return [
            self._pack_chunk(
                object_ids[offset : offset + self.batch_capacity],
                frame_index=frame_index,
                direction=direction,
                pointer_tpos_denominator=pointer_tpos_denominator,
            )
            for offset in range(0, len(object_ids), self.batch_capacity)
        ]

    def _pack_chunk(
        self,
        object_ids: list[int],
        *,
        frame_index: int,
        direction: int,
        pointer_tpos_denominator: int,
    ) -> PackedObjectState:
        capacity = self.batch_capacity
        padded_ids: list[int | None] = [*object_ids]
        padded_ids.extend([None] * (capacity - len(padded_ids)))
        object_valid = np.asarray(
            [value is not None for value in padded_ids], dtype=np.bool_
        )
        memory_valid = np.zeros((capacity, SPATIAL_CAPACITY), dtype=np.bool_)
        memory_age = np.zeros((capacity, SPATIAL_CAPACITY), dtype=np.int64)
        memory_conditioning = np.zeros((capacity, SPATIAL_CAPACITY), dtype=np.bool_)
        pointer_valid = np.zeros((capacity, POINTER_CAPACITY), dtype=np.bool_)
        pointer_age = np.zeros((capacity, POINTER_CAPACITY), dtype=np.int64)
        pointer_conditioning = np.zeros((capacity, POINTER_CAPACITY), dtype=np.bool_)
        pointer_denominator = np.full(
            (capacity,), pointer_tpos_denominator, dtype=np.float32
        )
        spatial_rows: list[tuple[VideoStateEntry | None, ...]] = []
        pointer_rows: list[tuple[VideoStateEntry | None, ...]] = []

        for row, object_id in enumerate(padded_ids):
            if object_id is None:
                spatial_rows.append((None,) * SPATIAL_CAPACITY)
                pointer_rows.append((None,) * POINTER_CAPACITY)
                continue
            state = self.objects[object_id]
            state.direction = direction
            selected_cond, unselected_cond = _closest_conditioning(
                {
                    index: entry
                    for index, entry in state.conditioning.items()
                    if index != frame_index
                },
                frame_index,
            )
            spatial: list[VideoStateEntry | None] = list(selected_cond)
            spatial.extend([None] * (CONDITIONING_CAPACITY - len(spatial)))
            non_cond: list[VideoStateEntry | None] = []
            for distance in range(1, NON_CONDITIONING_CAPACITY + 1):
                candidate = frame_index - direction * distance
                non_cond.append(
                    state.non_conditioning.get(candidate)
                    or unselected_cond.get(candidate)
                )
            spatial.extend(non_cond)
            spatial_rows.append(tuple(spatial))
            for column, entry in enumerate(spatial):
                if entry is None:
                    continue
                memory_valid[row, column] = True
                memory_age[row, column] = frame_index - entry.frame_index
                memory_conditioning[row, column] = entry.conditioning

            allowed_cond = [
                entry
                for entry in selected_cond
                if (
                    entry.frame_index >= frame_index
                    if direction < 0
                    else entry.frame_index <= frame_index
                )
            ]
            pointers: list[VideoStateEntry] = list(allowed_cond)
            for distance in range(1, POINTER_CAPACITY):
                candidate = frame_index - direction * distance
                entry = state.non_conditioning.get(candidate) or unselected_cond.get(
                    candidate
                )
                if entry is not None:
                    pointers.append(entry)
                if len(pointers) >= POINTER_CAPACITY:
                    break
            pointer_row: list[VideoStateEntry | None] = pointers[:POINTER_CAPACITY]
            pointer_row.extend([None] * (POINTER_CAPACITY - len(pointer_row)))
            pointer_rows.append(tuple(pointer_row))
            for column, entry in enumerate(pointer_row):
                if entry is None:
                    continue
                pointer_valid[row, column] = True
                pointer_age[row, column] = frame_index - entry.frame_index
                pointer_conditioning[row, column] = entry.conditioning

        return PackedObjectState(
            object_ids=tuple(padded_ids),
            object_valid=object_valid,
            spatial_entries=tuple(spatial_rows),
            memory_valid=memory_valid,
            memory_age=memory_age,
            memory_conditioning=memory_conditioning,
            pointer_entries=tuple(pointer_rows),
            pointer_valid=pointer_valid,
            pointer_age=pointer_age,
            pointer_conditioning=pointer_conditioning,
            pointer_tpos_denominator=pointer_denominator,
        )


__all__ = [
    "BaseVideoStateV1",
    "BaseVideoVariantParameters",
    "CONDITIONING_CAPACITY",
    "NON_CONDITIONING_CAPACITY",
    "POINTER_CAPACITY",
    "PackedObjectState",
    "SPATIAL_CAPACITY",
    "STATE_ABI_VERSION",
    "StateCapacityError",
    "VideoStateEntry",
]
