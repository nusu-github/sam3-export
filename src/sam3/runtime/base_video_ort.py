"""ORT CUDA IOBinding adapter for the fixed M4 SAM3 base video plan."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sam3.runtime.base_video import _BackendCommit, _BackendPreviewBatch
from sam3.runtime.base_video_state import PackedObjectState
from sam3.runtime.manifest import CapabilityError, ResolvedPlan


class OrtCudaBaseVideoAdapter:
    """Resolve only the five reviewed M4 roles and keep large values on CUDA."""

    _FRAME_ROLE = "tracker-frame-encode"
    _MULTI_ROLE = "base-tracker-preview-multimask3"
    _SINGLE_ROLE = "base-tracker-preview-single1"
    _COMMIT_ROLE = "base-memory-commit"
    _FUSED_ROLE = "base-tracker-step-and-commit-single1"

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
        required = {
            self._FRAME_ROLE,
            self._MULTI_ROLE,
            self._SINGLE_ROLE,
            self._COMMIT_ROLE,
            self._FUSED_ROLE,
        }
        missing = sorted(required - set(self._artifacts))
        if missing:
            raise CapabilityError(f"base video artifacts are missing: {missing}")
        files = {
            record["id"]: plan.bundle_dir / record["path"]
            for record in plan.manifest["files"]
        }
        self._sessions: dict[str, Any] = {}
        for role in required:
            artifact = self._artifacts[role]
            session = ort.InferenceSession(
                str(files[artifact["entry_file_ref"]]),
                providers=["CUDAExecutionProvider"],
            )
            session.disable_fallback()
            if "CUDAExecutionProvider" not in session.get_providers():
                raise CapabilityError(f"CUDAExecutionProvider did not load for {role}")
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

        static = {
            str(item["name"]): item["value"]
            for item in plan.manifest["profile"]["static_values"]
        }
        self.batch_capacity = int(static["object-batch-capacity"])
        if self.batch_capacity not in (4, 8):
            raise CapabilityError("M4 object batch capacity must be B4 or B8")
        policies = {
            str(item["name"]): item["value"] for item in plan.manifest["policies"]
        }
        steady = policies.get("steady-state-cut")
        if steady not in ("fused", "split"):
            raise CapabilityError("M4 manifest must select fused or split steady state")
        self.fused_default = steady == "fused"
        self.counters = {
            "frame_encodes": 0,
            "preview_launches": 0,
            "tracker_launches": 0,
            "memory_encodes": 0,
            "memory_commits": 0,
            "session_launches": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "d2d_pack_bytes": 0,
        }

    def _upload(self, value: np.ndarray) -> Any:
        contiguous = np.ascontiguousarray(value)
        self.counters["h2d_bytes"] += contiguous.nbytes
        return self._ort.OrtValue.ortvalue_from_numpy(contiguous, "cuda", 0)

    def _from_torch(self, value: torch.Tensor) -> Any:
        return self._ort.OrtValue.from_dlpack(value)

    @staticmethod
    def _as_torch(value: Any) -> torch.Tensor:
        result = torch.from_dlpack(value)
        if result.device.type != "cuda":
            raise CapabilityError("M4 device handoff left CUDA")
        return result

    def _run(self, role: str, inputs: dict[str, Any]) -> dict[str, Any]:
        artifact = self._artifacts[role]
        names = {
            item["tensor_ref"]: item["backend_name"] for item in artifact["inputs"]
        }
        binding = self._sessions[role].io_binding()
        for tensor_ref, value in inputs.items():
            binding.bind_ortvalue_input(names[tensor_ref], value)
        for output in self._sessions[role].get_outputs():
            binding.bind_output(output.name, "cuda", 0)
        self._sessions[role].run_with_iobinding(binding)
        self.counters["session_launches"] += 1
        values = binding.get_outputs()
        if any(value.device_name() != "cuda" for value in values):
            raise CapabilityError(f"IOBinding output left CUDA for {role}")
        refs = [item["tensor_ref"] for item in artifact["outputs"]]
        return dict(zip(refs, values))

    def _slice_numpy(self, value: Any, row: int) -> np.ndarray:
        tensor = self._as_torch(value)[row : row + 1].contiguous()
        host = self._from_torch(tensor).numpy()
        self.counters["d2h_bytes"] += host.nbytes
        return host

    def _slice_device(self, value: Any, row: int) -> Any:
        tensor = self._as_torch(value)[row : row + 1].contiguous()
        self.counters["d2d_pack_bytes"] += tensor.numel() * tensor.element_size()
        return self._from_torch(tensor)

    def encode_frame(self, values: np.ndarray) -> object:
        self.counters["frame_encodes"] += 1
        return self._run(self._FRAME_ROLE, {"pixel-values": self._upload(values)})

    def _repeat_frame(self, frame_cache: object) -> dict[str, Any]:
        if not isinstance(frame_cache, dict):
            raise TypeError("invalid M4 frame cache")
        result: dict[str, Any] = {}
        for name in (
            "frame-image-embedding",
            "frame-image-position",
            "frame-high-res-0",
            "frame-high-res-1",
        ):
            source = self._as_torch(frame_cache[name])
            expanded = source.expand(self.batch_capacity, -1, -1, -1).contiguous()
            self.counters["d2d_pack_bytes"] += (
                expanded.numel() * expanded.element_size()
            )
            result[name] = self._from_torch(expanded)
        return result

    def _pack_state(self, packed: PackedObjectState) -> dict[str, Any]:
        batch = self.batch_capacity
        memory = torch.zeros(
            (batch, 10, 64, 72, 72),
            dtype=torch.float16,
            device="cuda",
        )
        position = torch.zeros_like(memory)
        pointers = torch.zeros((batch, 16, 256), dtype=torch.float16, device="cuda")
        for row, entries in enumerate(packed.spatial_entries):
            for column, entry in enumerate(entries):
                if entry is None:
                    continue
                memory[row, column].copy_(self._as_torch(entry.memory_features)[0])
                position[row, column].copy_(self._as_torch(entry.memory_position)[0])
        for row, entries in enumerate(packed.pointer_entries):
            for column, entry in enumerate(entries):
                if entry is None:
                    continue
                pointers[row, column].copy_(self._as_torch(entry.object_pointer)[0])
        self.counters["d2d_pack_bytes"] += sum(
            value.numel() * value.element_size()
            for value in (memory, position, pointers)
        )
        return {
            "object-valid": self._upload(packed.object_valid),
            "memory-features": self._from_torch(memory),
            "memory-position": self._from_torch(position),
            "memory-valid": self._upload(packed.memory_valid),
            "memory-age": self._upload(packed.memory_age),
            "memory-conditioning": self._upload(packed.memory_conditioning),
            "object-pointers": self._from_torch(pointers),
            "pointer-valid": self._upload(packed.pointer_valid),
            "pointer-age": self._upload(packed.pointer_age),
            "pointer-conditioning": self._upload(packed.pointer_conditioning),
        }

    def _pack_prompt(self, prompt_inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in prompt_inputs.items():
            repeats = (self.batch_capacity,) + (1,) * (value.ndim - 1)
            result[name] = self._upload(np.tile(value, repeats))
        return result

    def _preview_from_outputs(
        self, outputs: dict[str, Any], packed: PackedObjectState, *, policy: str
    ) -> _BackendPreviewBatch:
        low_ref = f"preview-low-res-{policy}"
        score_ref = f"preview-scores-{policy}"
        logits: list[np.ndarray | None] = []
        scores: list[np.ndarray | None] = []
        for row, valid in enumerate(packed.object_valid):
            if not valid:
                logits.append(None)
                scores.append(None)
                continue
            logits.append(
                self._slice_numpy(outputs[low_ref], row)[0].astype(np.float32)
            )
            scores.append(
                self._slice_numpy(outputs[score_ref], row)[0].astype(np.float32)
            )
        return _BackendPreviewBatch(
            low_res_logits=tuple(logits),
            scores=tuple(scores),
            commit_masks=outputs["preview-commit-mask"],
            object_pointers=outputs["preview-object-pointer"],
            object_scores=outputs["preview-object-score"],
        )

    def preview(
        self,
        frame_cache: object,
        packed: PackedObjectState,
        prompt_inputs: dict[str, np.ndarray],
        *,
        multimask: bool,
    ) -> _BackendPreviewBatch:
        role = self._MULTI_ROLE if multimask else self._SINGLE_ROLE
        inputs = self._repeat_frame(frame_cache)
        inputs.update(self._pack_state(packed))
        inputs.update(self._pack_prompt(prompt_inputs))
        outputs = self._run(role, inputs)
        self.counters["preview_launches"] += 1
        self.counters["tracker_launches"] += 1
        return self._preview_from_outputs(
            outputs, packed, policy="multimask3" if multimask else "single1"
        )

    def _commits_from_outputs(
        self,
        outputs: dict[str, Any],
        packed: PackedObjectState,
        preview: _BackendPreviewBatch,
    ) -> tuple[_BackendCommit | None, ...]:
        result: list[_BackendCommit | None] = []
        for row, valid in enumerate(packed.object_valid):
            if not valid:
                result.append(None)
                continue
            high_res = self._slice_numpy(preview.commit_masks, row)[0].astype(
                np.float32
            )
            object_score = float(
                self._slice_numpy(preview.object_scores, row).reshape(-1)[0]
            )
            result.append(
                _BackendCommit(
                    memory_features=self._slice_device(
                        outputs["committed-memory-features"], row
                    ),
                    memory_position=self._slice_device(
                        outputs["committed-memory-position"], row
                    ),
                    object_pointer=self._slice_device(preview.object_pointers, row),
                    high_res_logits=high_res,
                    object_score=object_score,
                )
            )
        return tuple(result)

    def commit(
        self,
        frame_cache: object,
        packed: PackedObjectState,
        preview: _BackendPreviewBatch,
        *,
        is_mask_from_points: bool,
    ) -> tuple[_BackendCommit | None, ...]:
        inputs = self._repeat_frame(frame_cache)
        is_points = packed.object_valid & bool(is_mask_from_points)
        outputs = self._run(
            self._COMMIT_ROLE,
            {
                "frame-image-embedding": inputs["frame-image-embedding"],
                "preview-commit-mask": preview.commit_masks,
                "preview-object-score": preview.object_scores,
                "is-mask-from-points": self._upload(is_points),
            },
        )
        self.counters["memory_encodes"] += 1
        self.counters["memory_commits"] += int(packed.object_valid.sum())
        return self._commits_from_outputs(outputs, packed, preview)

    def step_and_commit(
        self,
        frame_cache: object,
        packed: PackedObjectState,
        prompt_inputs: dict[str, np.ndarray],
    ) -> tuple[_BackendPreviewBatch, tuple[_BackendCommit | None, ...]]:
        inputs = self._repeat_frame(frame_cache)
        inputs.update(self._pack_state(packed))
        inputs.update(self._pack_prompt(prompt_inputs))
        outputs = self._run(self._FUSED_ROLE, inputs)
        preview = self._preview_from_outputs(outputs, packed, policy="single1")
        self.counters["preview_launches"] += 1
        self.counters["tracker_launches"] += 1
        self.counters["memory_encodes"] += 1
        self.counters["memory_commits"] += int(packed.object_valid.sum())
        return preview, self._commits_from_outputs(outputs, packed, preview)

    def close(self) -> None:
        self._sessions.clear()


__all__ = ["OrtCudaBaseVideoAdapter"]
