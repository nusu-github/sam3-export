"""ORT CUDA IOBinding adapter for SAM3.1 native Multiplex video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sam3.runtime.manifest import CapabilityError, ResolvedPlan
from sam3.runtime.multiplex_state import (
    BUCKET_CAPACITY,
    CONDITIONING_CAPACITY,
    NON_CONDITIONING_CAPACITY,
    POINTER_FRAME_CAPACITY,
    SPATIAL_CAPACITY,
    SlotAssignment,
)
from sam3.runtime.multiplex_video import (
    _MultiplexBackendFrame,
    _MultiplexBackendPreview,
)


@dataclass(frozen=True)
class _DevicePreview:
    low_res: Any
    commit_mask: Any
    pointer: Any
    object_score: Any
    scores: Any


@dataclass
class _BucketFrame:
    frame_index: int
    conditioning: bool
    low_res: Any
    high_res: Any
    pointers: Any
    object_scores: Any
    selected_scores: Any
    memory: Any
    memory_position: Any
    propagation_image: Any
    propagation_position: Any
    slot_validity: torch.Tensor
    conditioning_validity: torch.Tensor
    conditioning_slots: frozenset[tuple[int, int]]
    bucket_validity: torch.Tensor


class OrtCudaMultiplexVideoAdapter:
    """Keep native bucket memory and pointers on CUDA between public results."""

    _FRAME_ROLE = "multiplex-frame-encode"
    _MULTI_ROLE = "multiplex-interaction-preview-multimask3"
    _SINGLE_ROLE = "multiplex-interaction-preview-single1"
    _BUCKET_INPUTS = {
        "propagation": {
            "slot_validity",
            "memory_features",
            "memory_position",
            "memory_image_features",
            "memory_image_position",
            "memory_valid",
            "memory_age",
            "object_pointers",
            "pointer_valid",
            "pointer_age",
        },
        "memory-commit": {
            "bucket_masks",
            "object_score",
            "slot_validity",
            "conditioning_validity",
        },
    }

    def __init__(self, plan: ResolvedPlan) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise CapabilityError(
                "ONNX Runtime CUDA is not installed; install sam3[ort-cuda]"
            ) from exc
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise CapabilityError("CUDAExecutionProvider is unavailable")
        self._ort = ort
        self._artifacts = plan.artifacts_by_role
        policies = {
            str(item["name"]): item["value"] for item in plan.manifest["policies"]
        }
        dispatch = policies.get("bucket-dispatch")
        if dispatch not in ("bounded-dynamic", "fixed-one-two"):
            raise CapabilityError("M5 manifest has no reviewed bucket dispatch")
        self._dynamic = dispatch == "bounded-dynamic"
        required = {
            self._FRAME_ROLE,
            self._MULTI_ROLE,
            self._SINGLE_ROLE,
        }
        if self._dynamic:
            required.update(
                {
                    "multiplex-propagation",
                    "multiplex-memory-commit",
                    "multiplex-scatter-replace-commit",
                }
            )
        else:
            required.update(
                {
                    "multiplex-propagation-bucket1",
                    "multiplex-memory-commit-bucket1",
                    "multiplex-scatter-replace-commit-bucket1",
                }
            )
        missing = sorted(required - set(self._artifacts))
        if missing:
            raise CapabilityError(f"multiplex artifacts are missing: {missing}")
        files = {
            record["id"]: plan.bundle_dir / record["path"]
            for record in plan.manifest["files"]
        }
        self._model_paths = {
            role: files[self._artifacts[role]["entry_file_ref"]] for role in required
        }
        self._sessions: dict[str, Any] = {}
        self._slot_validity = torch.zeros(
            (2, BUCKET_CAPACITY), dtype=torch.uint8, device="cuda"
        )
        self._conditioning: dict[int, _BucketFrame] = {}
        self._non_conditioning: dict[int, _BucketFrame] = {}
        self.counters = {
            "frame_encodes": 0,
            "preview_launches": 0,
            "propagation_launches": 0,
            "memory_commits": 0,
            "scatter_commits": 0,
            "session_launches": 0,
            "session_loads": 0,
            "final_d2h_bytes": 0,
            "state_d2h_bytes": 0,
            "state_h2d_bytes": 0,
            "frame_h2d_bytes": 0,
            "prompt_h2d_bytes": 0,
            "control_h2d_bytes": 0,
            "d2d_pack_bytes": 0,
            "state_demuxes": 0,
            "state_remuxes": 0,
        }

    def _role(self, operation: str, bucket_count: int) -> str:
        if self._dynamic:
            return f"multiplex-{operation}"
        del bucket_count
        return f"multiplex-{operation}-bucket1"

    def _session(self, role: str) -> Any:
        existing = self._sessions.get(role)
        if existing is not None:
            return existing
        options = self._ort.SessionOptions()
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = self._ort.InferenceSession(
            str(self._model_paths[role]),
            sess_options=options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                        "use_ep_level_unified_stream": "1",
                    },
                )
            ],
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise CapabilityError(f"CUDAExecutionProvider did not load for {role}")
        artifact = self._artifacts[role]
        declared_inputs = {item["backend_name"] for item in artifact["inputs"]}
        declared_outputs = {item["backend_name"] for item in artifact["outputs"]}
        actual_inputs = {item.name for item in session.get_inputs()}
        actual_outputs = {item.name for item in session.get_outputs()}
        if declared_inputs != actual_inputs or declared_outputs != actual_outputs:
            raise CapabilityError(
                f"ORT binding mismatch for {role}: "
                f"inputs={sorted(actual_inputs)} outputs={sorted(actual_outputs)}"
            )
        self._sessions[role] = session
        self.counters["session_loads"] += 1
        return session

    def _upload(self, value: np.ndarray, *, category: str) -> Any:
        contiguous = np.ascontiguousarray(value)
        self.counters[f"{category}_h2d_bytes"] += contiguous.nbytes
        return self._ort.OrtValue.ortvalue_from_numpy(contiguous, "cuda", 0)

    def _from_torch(self, value: torch.Tensor) -> Any:
        return self._ort.OrtValue.from_dlpack(value.contiguous())

    @staticmethod
    def _as_torch(value: Any) -> torch.Tensor:
        result = torch.from_dlpack(value)
        if result.device.type != "cuda":
            raise CapabilityError("Multiplex device handoff left CUDA")
        return result

    def _clone(self, value: Any) -> Any:
        tensor = self._as_torch(value).clone()
        self.counters["d2d_pack_bytes"] += tensor.numel() * tensor.element_size()
        return self._from_torch(tensor)

    def _run(self, role: str, inputs: dict[str, Any]) -> dict[str, Any]:
        session = self._session(role)
        expected = {item.name: item.type for item in session.get_inputs()}
        if set(inputs) != set(expected):
            raise CapabilityError(
                f"ORT input mismatch for {role}: {sorted(inputs)} != {sorted(expected)}"
            )
        binding = session.io_binding()
        for name, value in inputs.items():
            if value.data_type() != expected[name]:
                raise CapabilityError(
                    f"ORT dtype mismatch for {role}/{name}: "
                    f"{value.data_type()} != {expected[name]}"
                )
            binding.bind_ortvalue_input(name, value)
        for output in session.get_outputs():
            binding.bind_output(output.name, "cuda", 0)
        session.run_with_iobinding(binding)
        self.counters["session_launches"] += 1
        values = binding.get_outputs()
        if any(value.device_name() != "cuda" for value in values):
            raise CapabilityError(f"IOBinding output left CUDA for {role}")
        return {
            output.name: value for output, value in zip(session.get_outputs(), values)
        }

    def _bucket_slice(self, value: Any, bucket: int) -> Any:
        tensor = self._as_torch(value)[bucket : bucket + 1]
        return self._from_torch(tensor.contiguous())

    def _join_bucket_outputs(self, values: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in values[0]:
            tensors = [self._as_torch(item[name]) for item in values]
            joined = torch.cat(tensors, dim=0)
            self.counters["d2d_pack_bytes"] += joined.numel() * joined.element_size()
            result[name] = self._from_torch(joined)
        return result

    def _run_bucket_operation(
        self,
        operation: str,
        bucket_count: int,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        if self._dynamic or bucket_count == 1:
            return self._run(self._role(operation, bucket_count), inputs)
        bucket_inputs = self._BUCKET_INPUTS[operation]
        joined: dict[str, torch.Tensor] = {}
        for bucket in range(bucket_count):
            values = {
                name: (
                    (
                        self._from_torch(self._as_torch(value)[bucket].contiguous())
                        if name
                        in {
                            "memory_image_features",
                            "memory_image_position",
                        }
                        else self._bucket_slice(value, bucket)
                    )
                    if name in bucket_inputs
                    else value
                )
                for name, value in inputs.items()
            }
            outputs = self._run(self._role(operation, 1), values)
            for name, value in outputs.items():
                tensor = self._as_torch(value)
                if name not in joined:
                    joined[name] = torch.empty(
                        (bucket_count, *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                joined[name][bucket : bucket + 1].copy_(tensor)
        # ORT may reuse its output arena on the next invocation.  The slices
        # above therefore materialize the reviewed two-bucket state directly
        # on CUDA before the following bucket runs.
        self._sessions.pop(self._role(operation, 1), None)
        self.counters["d2d_pack_bytes"] += sum(
            value.numel() * value.element_size() for value in joined.values()
        )
        return {name: self._from_torch(value) for name, value in joined.items()}

    def _run_scatter(
        self,
        bucket_count: int,
        inputs: dict[str, Any],
        assignment: SlotAssignment,
        previous_memory: Any | None = None,
        previous_memory_position: Any | None = None,
    ) -> dict[str, Any]:
        if self._dynamic or bucket_count == 1:
            return self._run(self._role("scatter-replace-commit", bucket_count), inputs)
        if previous_memory is None or previous_memory_position is None:
            raise CapabilityError(
                "two-bucket scatter requires the existing bucket memory"
            )
        bucket_inputs = {
            "bucket_low_res",
            "bucket_high_res",
            "bucket_pointers",
            "bucket_object_scores",
            "slot_validity",
            "conditioning_validity",
        }
        outputs = []
        for bucket in range(bucket_count):
            if bucket != assignment.bucket:
                outputs.append(
                    {
                        "bucket_low_res_out": self._bucket_slice(
                            inputs["bucket_low_res"], bucket
                        ),
                        "bucket_high_res_out": self._bucket_slice(
                            inputs["bucket_high_res"], bucket
                        ),
                        "bucket_pointers_out": self._bucket_slice(
                            inputs["bucket_pointers"], bucket
                        ),
                        "bucket_object_scores_out": self._bucket_slice(
                            inputs["bucket_object_scores"], bucket
                        ),
                        "memory_features": self._bucket_slice(previous_memory, bucket),
                        "memory_position": self._bucket_slice(
                            previous_memory_position, bucket
                        ),
                    }
                )
                continue
            values = {
                name: (
                    self._bucket_slice(value, bucket)
                    if name in bucket_inputs
                    else value
                )
                for name, value in inputs.items()
                if name != "assignment"
            }
            local_assignment = torch.tensor(
                [[0, assignment.slot]], dtype=torch.int64, device="cuda"
            )
            values["assignment"] = self._from_torch(local_assignment)
            outputs.append(self._run(self._role("scatter-replace-commit", 1), values))
        return self._join_bucket_outputs(outputs)

    def _to_public_numpy(self, value: Any) -> np.ndarray:
        host = value.numpy()
        self.counters["final_d2h_bytes"] += host.nbytes
        return host

    def _bucket_count(self) -> int:
        occupied = torch.any(self._slot_validity, dim=1)
        if bool(occupied[1]):
            return 2
        return 1

    def reset_state(self) -> None:
        self._slot_validity.zero_()
        self._conditioning.clear()
        self._non_conditioning.clear()

    def add_slot(self, assignment: SlotAssignment) -> None:
        self._slot_validity[assignment.bucket, assignment.slot] = True
        self.counters["control_h2d_bytes"] += 1

    def remove_slot(self, assignment: SlotAssignment) -> None:
        self._slot_validity[assignment.bucket, assignment.slot] = False
        self.counters["control_h2d_bytes"] += 1
        for entry in [*self._conditioning.values(), *self._non_conditioning.values()]:
            if assignment.bucket >= entry.slot_validity.shape[0]:
                continue
            entry.slot_validity[assignment.bucket, assignment.slot] = False
            entry.conditioning_validity[assignment.bucket, assignment.slot] = False
            entry.conditioning_slots = entry.conditioning_slots - {
                (assignment.bucket, assignment.slot)
            }
            for value, fill in (
                (entry.low_res, -1024.0),
                (entry.high_res, -1024.0),
                (entry.pointers, 0.0),
                (entry.object_scores, -1024.0),
                (entry.selected_scores, 0.0),
            ):
                tensor = self._as_torch(value)
                tensor[assignment.bucket, assignment.slot] = fill
            entry.bucket_validity = torch.any(entry.slot_validity, dim=1)
            outputs = self._run_bucket_operation(
                "memory-commit",
                entry.slot_validity.shape[0],
                {
                    "propagation_image": entry.propagation_image,
                    "bucket_masks": entry.high_res,
                    "object_score": entry.object_scores,
                    "slot_validity": self._from_torch(entry.slot_validity),
                    "conditioning_validity": self._from_torch(
                        entry.conditioning_validity
                    ),
                },
            )
            entry.memory = outputs["memory_features"]
            entry.memory_position = outputs["memory_position"]
            self.counters["memory_commits"] += 1

    def encode_frame(self, values: np.ndarray) -> object:
        outputs = self._run(
            self._FRAME_ROLE,
            {"pixel_values": self._upload(values, category="frame")},
        )
        self.counters["frame_encodes"] += 1
        return {name: self._clone(value) for name, value in outputs.items()}

    def preview(
        self,
        frame_cache: object,
        assignment: SlotAssignment,
        prompt_inputs: dict[str, np.ndarray],
        *,
        multimask: bool,
    ) -> _MultiplexBackendPreview:
        del assignment
        if not isinstance(frame_cache, dict):
            raise TypeError("invalid Multiplex frame cache")
        role = self._MULTI_ROLE if multimask else self._SINGLE_ROLE
        inputs = {
            "interactive_image": frame_cache["interactive_image"],
            "interactive_high_res_0": frame_cache["interactive_high_res_0"],
            "interactive_high_res_1": frame_cache["interactive_high_res_1"],
        }
        inputs.update(
            {
                name.replace("-", "_"): self._upload(value, category="prompt")
                for name, value in prompt_inputs.items()
            }
        )
        outputs = self._run(role, inputs)
        self.counters["preview_launches"] += 1
        device = _DevicePreview(
            low_res=outputs["low_res_logits"],
            commit_mask=outputs["commit_mask"],
            pointer=outputs["object_pointer"],
            object_score=outputs["object_score"],
            scores=outputs["scores"],
        )
        return _MultiplexBackendPreview(
            low_res_logits=self._to_public_numpy(outputs["low_res_logits"])[0],
            scores=self._to_public_numpy(outputs["scores"])[0],
            commit_mask=device,
            object_pointer=device,
            object_score=device,
        )

    def _empty_bucket_values(self, bucket_count: int) -> tuple[Any, Any, Any, Any, Any]:
        low = torch.full(
            (bucket_count, BUCKET_CAPACITY, 1, 288, 288),
            -1024.0,
            dtype=torch.float32,
            device="cuda",
        )
        high = torch.full(
            (bucket_count, BUCKET_CAPACITY, 1, 1008, 1008),
            -1024.0,
            dtype=torch.float32,
            device="cuda",
        )
        pointers = torch.zeros(
            (bucket_count, BUCKET_CAPACITY, 256),
            dtype=torch.float16,
            device="cuda",
        )
        object_scores = torch.full(
            (bucket_count, BUCKET_CAPACITY, 1),
            -1024.0,
            dtype=torch.float32,
            device="cuda",
        )
        scores = torch.zeros(
            (bucket_count, BUCKET_CAPACITY),
            dtype=torch.float32,
            device="cuda",
        )
        return tuple(
            self._from_torch(value)
            for value in (low, high, pointers, object_scores, scores)
        )

    def _resize_buckets(self, value: Any, bucket_count: int, fill: float) -> Any:
        tensor = self._as_torch(value)
        if tensor.shape[0] == bucket_count:
            return value
        if tensor.shape[0] > bucket_count:
            result = tensor[:bucket_count].clone()
        else:
            result = torch.full(
                (bucket_count, *tensor.shape[1:]),
                fill,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            result[: tensor.shape[0]].copy_(tensor)
        self.counters["d2d_pack_bytes"] += result.numel() * result.element_size()
        return self._from_torch(result)

    def commit(
        self,
        frame_cache: object,
        assignment: SlotAssignment,
        preview: _MultiplexBackendPreview,
        *,
        frame_index: int,
    ) -> None:
        if not isinstance(frame_cache, dict):
            raise TypeError("invalid Multiplex frame cache")
        if not isinstance(preview.commit_mask, _DevicePreview):
            raise TypeError("invalid Multiplex device preview")
        device = preview.commit_mask
        bucket_count = self._bucket_count()
        existing = self._conditioning.get(frame_index) or self._non_conditioning.get(
            frame_index
        )
        if existing is None:
            low, high, pointers, object_scores, selected_scores = (
                self._empty_bucket_values(bucket_count)
            )
            conditioning_validity = torch.zeros(
                (bucket_count, BUCKET_CAPACITY),
                dtype=torch.uint8,
                device="cuda",
            )
        else:
            low = self._resize_buckets(existing.low_res, bucket_count, -1024.0)
            high = self._resize_buckets(existing.high_res, bucket_count, -1024.0)
            pointers = self._resize_buckets(existing.pointers, bucket_count, 0.0)
            object_scores = self._resize_buckets(
                existing.object_scores, bucket_count, -1024.0
            )
            selected_scores = self._resize_buckets(
                existing.selected_scores, bucket_count, 0.0
            )
            conditioning_validity = torch.zeros(
                (bucket_count, BUCKET_CAPACITY),
                dtype=torch.uint8,
                device="cuda",
            )
            previous = existing.conditioning_validity
            conditioning_validity[: previous.shape[0]].copy_(previous)
        previous_memory = (
            None
            if existing is None
            else self._resize_buckets(existing.memory, bucket_count, 0.0)
        )
        previous_memory_position = (
            None
            if existing is None
            else self._resize_buckets(existing.memory_position, bucket_count, 0.0)
        )
        conditioning_slots = (
            set() if existing is None else set(existing.conditioning_slots)
        )
        conditioning_slots.add((assignment.bucket, assignment.slot))
        conditioning_validity[assignment.bucket, assignment.slot] = True
        assignment_tensor = torch.tensor(
            [[assignment.bucket, assignment.slot]],
            dtype=torch.int64,
            device="cuda",
        )
        self.counters["control_h2d_bytes"] += assignment_tensor.numel() * 8
        validity = self._slot_validity[:bucket_count].clone()
        outputs = self._run_scatter(
            bucket_count,
            {
                "propagation_image": frame_cache["propagation_image"],
                "bucket_low_res": low,
                "bucket_high_res": high,
                "bucket_pointers": pointers,
                "bucket_object_scores": object_scores,
                "replacement_low_res": device.low_res,
                "replacement_high_res": device.commit_mask,
                "replacement_pointer": device.pointer,
                "replacement_object_score": device.object_score,
                "assignment": self._from_torch(assignment_tensor),
                "slot_validity": self._from_torch(validity),
                "conditioning_validity": self._from_torch(conditioning_validity),
            },
            assignment,
            previous_memory,
            previous_memory_position,
        )
        selected = self._as_torch(selected_scores)
        selected[assignment.bucket, assignment.slot] = self._as_torch(device.scores)[
            0, 0
        ]
        entry = _BucketFrame(
            frame_index=frame_index,
            conditioning=True,
            low_res=outputs["bucket_low_res_out"],
            high_res=outputs["bucket_high_res_out"],
            pointers=outputs["bucket_pointers_out"],
            object_scores=outputs["bucket_object_scores_out"],
            selected_scores=selected_scores,
            memory=outputs["memory_features"],
            memory_position=outputs["memory_position"],
            propagation_image=frame_cache["propagation_image"],
            propagation_position=frame_cache["propagation_position"],
            slot_validity=validity,
            conditioning_validity=conditioning_validity,
            conditioning_slots=frozenset(conditioning_slots),
            bucket_validity=torch.any(validity, dim=1),
        )
        self._conditioning[frame_index] = entry
        self._non_conditioning.pop(frame_index, None)
        self.counters["scatter_commits"] += 1
        self.counters["memory_commits"] += 1

    @staticmethod
    def _closest_conditioning(
        entries: dict[int, _BucketFrame], frame_index: int
    ) -> tuple[list[_BucketFrame], dict[int, _BucketFrame]]:
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
        selected_frames.update(
            remaining[: CONDITIONING_CAPACITY - len(selected_frames)]
        )
        return (
            [entries[value] for value in entries if value in selected_frames],
            {
                value: entry
                for value, entry in entries.items()
                if value not in selected_frames
            },
        )

    def _pack_state(
        self, frame_index: int, reverse: bool, bucket_count: int
    ) -> dict[str, Any]:
        direction = -1 if reverse else 1
        device = torch.device("cuda")
        memory = torch.zeros(
            (bucket_count, SPATIAL_CAPACITY, 256, 72, 72),
            dtype=torch.float16,
            device=device,
        )
        memory_position = torch.zeros_like(memory)
        memory_image = torch.zeros(
            (bucket_count, SPATIAL_CAPACITY, 256, 72, 72),
            dtype=torch.float16,
            device=device,
        )
        memory_image_position = torch.zeros_like(memory_image)
        memory_valid = torch.zeros(
            (bucket_count, SPATIAL_CAPACITY), dtype=torch.uint8, device=device
        )
        memory_age = torch.zeros(
            (bucket_count, SPATIAL_CAPACITY), dtype=torch.int64, device=device
        )
        object_pointers = torch.zeros(
            (bucket_count, POINTER_FRAME_CAPACITY, BUCKET_CAPACITY, 256),
            dtype=torch.float16,
            device=device,
        )
        pointer_valid = torch.zeros(
            (bucket_count, POINTER_FRAME_CAPACITY, BUCKET_CAPACITY),
            dtype=torch.uint8,
            device=device,
        )
        pointer_age = torch.zeros(
            (bucket_count, POINTER_FRAME_CAPACITY),
            dtype=torch.int64,
            device=device,
        )

        for bucket in range(bucket_count):

            def is_conditioning(entry: _BucketFrame) -> bool:
                return entry.conditioning and any(
                    selected_bucket == bucket
                    for selected_bucket, unused_slot in entry.conditioning_slots
                )

            conditioning = {
                index: entry
                for index, entry in self._conditioning.items()
                if index != frame_index and is_conditioning(entry)
            }
            non_conditioning = dict(self._non_conditioning)
            non_conditioning.update(
                {
                    index: entry
                    for index, entry in self._conditioning.items()
                    if index != frame_index and not is_conditioning(entry)
                }
            )
            selected_cond, unselected_cond = self._closest_conditioning(
                conditioning, frame_index
            )
            spatial: list[_BucketFrame | None] = list(selected_cond)
            spatial_tpos = [
                abs(frame_index - entry.frame_index) for entry in selected_cond
            ]
            spatial.extend([None] * (CONDITIONING_CAPACITY - len(spatial)))
            spatial_tpos.extend([0] * (CONDITIONING_CAPACITY - len(spatial_tpos)))
            for distance in range(1, NON_CONDITIONING_CAPACITY + 1):
                candidate = frame_index - direction * distance
                spatial.append(
                    non_conditioning.get(candidate) or unselected_cond.get(candidate)
                )
                spatial_tpos.append(NON_CONDITIONING_CAPACITY + 1 - distance)
            for column, (entry, temporal_position) in enumerate(
                zip(spatial, spatial_tpos)
            ):
                if (
                    entry is None
                    or bucket >= entry.bucket_validity.shape[0]
                    or not bool(entry.bucket_validity[bucket].item())
                ):
                    continue
                memory[bucket, column].copy_(self._as_torch(entry.memory)[bucket])
                memory_position[bucket, column].copy_(
                    self._as_torch(entry.memory_position)[bucket]
                )
                memory_image[bucket, column].copy_(
                    self._as_torch(entry.propagation_image)[0]
                )
                memory_image_position[bucket, column].copy_(
                    self._as_torch(entry.propagation_position)[0]
                )
                memory_valid[bucket, column] = 1
                memory_age[bucket, column] = temporal_position

            allowed_cond = [
                entry
                for entry in selected_cond
                if (
                    entry.frame_index >= frame_index
                    if reverse
                    else entry.frame_index <= frame_index
                )
            ]
            pointers: list[_BucketFrame] = list(allowed_cond)
            for distance in range(1, POINTER_FRAME_CAPACITY):
                candidate = frame_index - direction * distance
                entry = non_conditioning.get(candidate) or unselected_cond.get(
                    candidate
                )
                if entry is not None:
                    pointers.append(entry)
                if len(pointers) >= POINTER_FRAME_CAPACITY:
                    break
            for column, entry in enumerate(pointers[:POINTER_FRAME_CAPACITY]):
                if bucket >= entry.slot_validity.shape[0]:
                    continue
                object_pointers[bucket, column].copy_(
                    self._as_torch(entry.pointers)[bucket]
                )
                pointer_valid[bucket, column].copy_(entry.slot_validity[bucket])
                pointer_age[bucket, column] = frame_index - entry.frame_index
        memory_image_value = memory_image[0] if bucket_count == 1 else memory_image
        memory_image_position_value = (
            memory_image_position[0] if bucket_count == 1 else memory_image_position
        )
        packed = {
            "memory_features": memory,
            "memory_position": memory_position,
            "memory_image_features": memory_image_value,
            "memory_image_position": memory_image_position_value,
            "memory_valid": memory_valid,
            "memory_age": memory_age,
            "object_pointers": object_pointers,
            "pointer_valid": pointer_valid,
            "pointer_age": pointer_age,
        }
        self.counters["d2d_pack_bytes"] += sum(
            value.numel() * value.element_size() for value in packed.values()
        )
        return {name: self._from_torch(value) for name, value in packed.items()}

    def _public_frame(
        self, entry: _BucketFrame, assignments: np.ndarray
    ) -> _MultiplexBackendFrame:
        location = torch.as_tensor(assignments, dtype=torch.int64, device="cuda")
        self.counters["control_h2d_bytes"] += location.numel() * 8
        low = self._as_torch(entry.low_res)[
            location[:, 0], location[:, 1], 0
        ].contiguous()
        scores = self._as_torch(entry.selected_scores)[
            location[:, 0], location[:, 1]
        ].contiguous()
        self.counters["d2d_pack_bytes"] += (
            low.numel() * low.element_size() + scores.numel() * scores.element_size()
        )
        return _MultiplexBackendFrame(
            low_res_logits=self._to_public_numpy(self._from_torch(low)),
            scores=self._to_public_numpy(self._from_torch(scores)),
        )

    def propagate(
        self,
        frame_cache: object,
        assignments: np.ndarray,
        *,
        frame_index: int,
        reverse: bool,
    ) -> _MultiplexBackendFrame:
        if not isinstance(frame_cache, dict):
            raise TypeError("invalid Multiplex frame cache")
        existing = self._conditioning.get(frame_index) or self._non_conditioning.get(
            frame_index
        )
        if existing is not None:
            return self._public_frame(existing, assignments)
        bucket_count = self._bucket_count()
        inputs = {
            "propagation_image": frame_cache["propagation_image"],
            "propagation_position": frame_cache["propagation_position"],
            "propagation_high_res_0": frame_cache["propagation_high_res_0"],
            "propagation_high_res_1": frame_cache["propagation_high_res_1"],
            "slot_validity": self._from_torch(
                self._slot_validity[:bucket_count].clone()
            ),
            **self._pack_state(frame_index, reverse, bucket_count),
        }
        outputs = self._run_bucket_operation("propagation", bucket_count, inputs)
        self.counters["propagation_launches"] += 1
        validity = self._slot_validity[:bucket_count].clone()
        conditioning_validity = torch.zeros_like(validity)
        committed = self._run_bucket_operation(
            "memory-commit",
            bucket_count,
            {
                "propagation_image": frame_cache["propagation_image"],
                "bucket_masks": outputs["selected_high_res"],
                "object_score": outputs["object_score"],
                "slot_validity": self._from_torch(validity),
                "conditioning_validity": self._from_torch(conditioning_validity),
            },
        )
        self.counters["memory_commits"] += 1
        selected_scores = self._as_torch(outputs["scores"]).max(dim=2).values
        entry = _BucketFrame(
            frame_index=frame_index,
            conditioning=False,
            low_res=outputs["selected_low_res"],
            high_res=outputs["selected_high_res"],
            pointers=outputs["object_pointers_out"],
            object_scores=outputs["object_score"],
            selected_scores=self._from_torch(selected_scores),
            memory=committed["memory_features"],
            memory_position=committed["memory_position"],
            propagation_image=frame_cache["propagation_image"],
            propagation_position=frame_cache["propagation_position"],
            slot_validity=validity,
            conditioning_validity=conditioning_validity,
            conditioning_slots=frozenset(),
            bucket_validity=torch.any(validity, dim=1),
        )
        self._non_conditioning[frame_index] = entry
        return self._public_frame(entry, assignments)

    def close(self) -> None:
        self.reset_state()
        self._sessions.clear()


__all__ = ["OrtCudaMultiplexVideoAdapter"]
