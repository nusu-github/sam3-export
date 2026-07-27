# M6 backend bundles: direct ExportedProgram adopted as candidate

Date / owner: 2026-07-27 / Export Tech Lead + Runtime Lead

## Decision

Keep all existing ORT CUDA dispatch roles unchanged. Admit direct PyTorch
`ExportedProgram` CUDA execution for the M2 fused image PCS plan as an
**evaluated optional candidate**, not as the default. Reject the evaluated
AOTInductor package for dispatch because end-to-end admitted-query parity
fails, even though all three roles compile.

M3–M5 receive canonical saved captures in M6, but non-ONNX execution for those
plans remains not evaluated. No result in this record combines SAM3 base state
with SAM3.1 Multiplex state.

Applicable profiles:

- `sam3_base_image_pcs_text_ortcuda_v1`,
  `b1-1008-l32-q200-fp16`, for the backend comparison.
- M2–M5 shipped profiles for canonical capture metadata and save/load gates.

## Fixed comparison conditions

Both non-ONNX runs consume the same saved M2 `ExportedProgram` files and the
same five owned image/text fixtures as the ORT fused plan. The Public API,
strict `score > 0.5` policy, stable query ordering, warmup 2 and repeats 5 are
fixed. Environment: Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, NVIDIA RTX
4000 Ada Generation (20,475 MiB), driver 580.65.06. ORT is 1.27.0.

## Backend matrix and measurements

| Backend | Support / fallback | Task parity | Artifact bytes | Median / p95 | Peak / persistent VRAM | Decision |
|---|---|---|---:|---:|---:|---|
| ORT CUDA | CUDA EP + IOBinding; fused default and explicit corrected-split fallback both validated | all five exact admitted-index gates pass; fused minimum mask IoU 0.9957 | recorded per M2 manifest/report | M1/M2 report-owned | M1/M2 report-owned | shipped roles unchanged |
| direct `ExportedProgram` CUDA | PyTorch ATen CUDA; no backend fallback or semantic-ABI change | all five exact; minimum mask IoU 0.9952 | 1,769,347,290 | 87.49 / 97.35 ms | 7,844,236,288 / 6,972,286,976 bytes | optional candidate |
| AOTInductor CUDA | all 3 roles package; no unsupported-op compile error or reported compiler fallback node | 3/5 fixtures fail exact admitted indices | 1,693,062,125 | 104.22 / 134.99 ms | 249,114,112 / 66,934,784 bytes | rejected |

The AOTInductor VRAM counters describe allocations visible during the timed
post-load interval and are therefore not directly interchangeable with direct
`ExportedProgram` persistent module storage. They are recorded, not used to
override the parity gate.

Representative AOT failures at the fixed threshold include
`truck_single` expected `[144]` versus nine different admitted queries,
`truck_multiple` expected four versus 36, and `groceries_broad` expected two
versus none. Thresholds were not relaxed and the backend difference was not
made into an API difference.

## Release checklist

- [x] Every public M2–M5 generated manifest owns capture mode, graph signature,
  range constraints, fixture hash and saved-program file references.
- [x] Saved programs are decomposed to an ATen capture and immediately reloaded;
  non-contiguous constants are normalized before serialization.
- [x] M2, M3, M4 and M5 ORT CUDA release validators pass with their existing
  Public API and device-resident handoff gates.
- [x] The M2 direct `ExportedProgram` bundle passes the same semantic Public API
  fixture gate.
- [x] AOTInductor compile support, fallback evidence, size, latency, VRAM and
  parity failure are recorded.
- [x] Default/optional/fallback roles are explicit; no default changed.
