# M4 — SAM3 base video batch and propagation cut

## Decision

Ship plan `sam3_base_video_tracking_ortcuda_v1` with the B4 object profile and
`BaseTrackerStepAndCommitSingle1` as the steady-state propagation artifact.
Keep correction as non-mutating preview followed by an explicit final
`BaseMemoryCommit`. B8 and split steady-state remain measured candidates in the
ignored work area and are not included as fallback plans.

## Applicable profiles

- SAM3 base video tracking with point, box and prior-mask correction,
  per-object batching and `BaseVideoStateV1`.
- `b4-1008-p16-box1-mask288-m10-ptr16-fp16`: image 1008, prompt capacity
  P16/box1/mask288, 10 spatial memories, 16 pointers and FP16.
- ONNX opset 18, ONNX Runtime 1.27.0 CUDA EP and CUDA IOBinding on NVIDIA RTX
  4000 Ada.

The decision does not apply to M2/M3 image plans, other shapes/precisions or
SAM3.1 Tri neck, bucket state and Multiplex.

## Batch evidence

Both candidates used the same fixture, dtype, two warmups, five repeats,
output policy and CUDA-resident handoff.

| Candidate | Median / p95 (ms) | Peak / persistent VRAM (bytes) | D2H / H2D |
|---|---:|---:|---:|
| B4 | 284.273 / 285.006 | 2,710,011,904 / 1,212,967,424 | 0 / 0 |
| B8 | 570.353 / 570.467 | 4,285,731,328 / 1,347,023,360 | 0 / 0 |

B8 was 0.318% slower than two B4 launches, rather than at least 15% faster.
Its peak/B4 ratio was 1.581, within the 1.75 memory ceiling, but the latency
gate failed; B4 is therefore selected. Public validation confirmed tracker
launch counts of 1, 1 and 2 for 1, 4 and 5 objects respectively, with one
frame encode and one commit per active object.

## Fused-cut evidence

| Candidate | Median / p95 (ms) | Peak / persistent VRAM (bytes) | Intermediate copies | Launches |
|---|---:|---:|---:|---:|
| split preview + commit | 295.177 / 295.934 | 2,710,536,192 / 1,212,967,424 | 0 D2H / 0 H2D | 2 |
| fused step + commit | 295.467 / 296.813 | 2,710,536,192 / 1,212,967,424 | 0 D2H / 0 H2D | 1 |

The fused/split median ratio was 1.00098 and the peak ratio was 1.0, within
the 5% latency and 1.25x memory gates. Prediction/memory parity and CUDA
residency also passed, so fused is the shipped steady-state recipe. Correction
retains the split policy boundary because the user may preview repeatedly and
commit only the final single result.

## Parity and residency

The owned trajectory compared official eager, local eager, canonical
`ExportedProgram`, ORT CUDA and Public API for memory 0/1/max, repeated
correction, forward/reverse propagation, replacement, object absence and
batch/chunk isolation. Public ORT task-mask IoU was 0.999678–0.999880 for
memory 0/1/max, score maximum absolute difference was at most 0.000488 and
low-resolution-logit MAE was at most 0.027849. The max trajectory packed 4
conditioning + 6 non-conditioning spatial entries and all 16 pointer slots.

Frame, memory, pointer and preview-to-commit values stayed in CUDA
`OrtValue`/DLPack storage. Fixed prompt/state values crossed H2D and D2H was
limited to public scores, final logits and final masks. CUDA EP fallback was
disabled and the plan declares no fallback.

## Replay

Fixture: `tests/fixtures/m4_base_video/cases.json`
(`m4-base-video-trajectory-v1`), warmup 2, five repeats. Official SAM3 commit:
`cdff5a9`; checkpoint SHA-256:
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

```bash
PYTHONPATH=src python scripts/export_base_video_v2.py \
  --official-repo ../sam3 --checkpoint /path/to/sam3.pt
```

The selected graph/external-data pairs total 1,129,918,974 bytes. The ignored
bundle owns exact per-file sizes/hashes, graph signatures,
decision JSON, provenance, environment and copy/launch counters. Candidate
work remains under `.m4-work/` and is not part of the selected bundle.
