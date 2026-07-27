# M5 SAM3.1 Multiplex profile decision

This record owns the M5 fixed-versus-bounded-dynamic profile decision. Detailed
per-stage numerical results and package hashes remain in the generated release
reports rather than being duplicated here.

## Decision

Adopt the fixed one-bucket artifact and dispatch it once for 1–16 objects or
twice for 17–32 objects. Each dispatch operates on one native 16-slot
`MultiplexStateV1` bucket. The two-bucket path composes two independent
one-bucket trajectories and does not demux or remux large state through the
host.

The bounded-dynamic bucket-count candidate is rejected. A separately captured
fixed B2 artifact is also rejected; it is not shipped as a fallback.

## Applicable profiles

- `fixed-bucket1-dispatch1to2-1008-p16-mask288-m10-ptr16-fp16`
- Plan: `sam3_1_multiplex_video_tracking_ortcuda_v1`
- Backend: ONNX Runtime 1.27.0 CUDA EP with CUDA IOBinding
- Public capacity: 16 slots per bucket, one or two buckets, at most 32 objects

This decision does not apply to SAM3 base video, CPU execution, unbounded
bucket counts or M6 backend bundles.

## Reference and protocol

| Item | Recorded value |
|---|---|
| Model revision | `daa63191845a41281374e725f4c9e51c7a824460` |
| Official source commit | `cdff5a927beba49b6249c1e2973b29bd64f40b83` |
| Checkpoint | `sam3.1_multiplex.pt` |
| Checkpoint SHA-256 | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` |
| Export implementation commit | `14260a309d821c751ce42b627109f155ea776030` |
| Device | NVIDIA RTX 4000 Ada Generation 20 GB |
| Software | PyTorch `2.13.0+cu130`; ONNX Runtime `1.27.0` |
| Measurement | FP16, warmup 2, repeats 5, CUDA IOBinding |

The checkpoint adapter mapped 474 Tri-neck parameters and 457 native
Multiplex tracker parameters with no missing or unexpected keys. It did not
reuse M4 memory constants, the dual/SAM2 neck, or per-object pointer layout.

## Measurements

The fixed measurements below use the same fixture and handoff conditions.
VRAM is the measured peak/persistent allocation for the loaded fixed recipe.
State copy bytes exclude the final public output boundary.

| Objects | Bucket dispatches | Median ms | p95 ms | Peak / persistent VRAM | State D2H / H2D | Launches / frame |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2077.720 | 2077.949 | 12,879,724,544 B | 0 / 0 B | 2 |
| 16 | 1 | 2077.788 | 2077.985 | 12,879,724,544 B | 0 / 0 B | 2 |
| 17 | 2 | 4159.567 | 4159.641 | 12,879,724,544 B | 0 / 0 B | 4 |

The compared fixed operation artifacts total 74,986,814 bytes. The
bounded-dynamic candidate totals 75,931,947 bytes, so it also fails the
requirement that its artifact size be no larger than the fixed recipe.

## Candidate disposition

| Candidate | Result | Reason |
|---|---|---|
| Fixed B1 dispatched 1–2 times | Adopt | CUDA-only, exact independent bucket composition, 0 large-state D2H/H2D |
| Fixed B2 artifact | Reject | 20 GB residency pressure and independent one-bucket trajectory parity failure |
| Bounded-dynamic bucket count 1–2 | Reject | ORT assigned nodes to CPU EP when fallback was disabled; artifact size gate also failed |

Because the bounded-dynamic candidate failed residency and size before meeting
the common gate, its latency and VRAM ratios are not used to claim adoption.
The rejected candidates are absent from the public bundle and are not
fallbacks.

## Parity and residency result

The release validator passed official eager → local eager →
`ExportedProgram` → ORT CUDA → Public API for 1/2/15/16 slots and 17/32
objects. Public frame-2 active-mask IoU ranged from 0.9954 to 0.9991; the
16/17 bucket-boundary IoUs were at least 0.9935 and 0.9993 respectively.
Scores, low-resolution logits, memory and pointers remain recorded in the
fixture report. Selected correction preserved every non-target bucket byte.

CUDA EP fallback-node count is zero for the adopted artifacts. Frame,
shared-memory and pointer state records 0 D2H, 0 H2D, 0 state demux and 0 state
remux during propagation. D2H occurs only when constructing public masks,
scores, low-resolution logits and metadata.

## Replay

Generate the official fixture, build the bundle atomically, then run the
release validator:

```bash
PYTHONPATH=src python scripts/m5_official_multiplex_reference.py \
  --official-repo ../sam3 \
  --checkpoint /path/to/sam3.1_multiplex.pt \
  --output-dir /tmp/m5-official
PYTHONPATH=src python scripts/export_multiplex_video_v2.py \
  --official-repo ../sam3 \
  --checkpoint /path/to/sam3.1_multiplex.pt \
  --official-reference-dir /tmp/m5-official
PYTHONPATH=src python scripts/validate_multiplex_video_v2.py \
  --bundle-dir artifacts/sam3-multiplex-video-tracking-ortcuda-v2
```

The machine-readable decision, full measurements, per-stage parity, graph
signatures and fixture/package hashes are in `reports/profile_decision.json`,
`reports/m5_release_validation.json`, `reports/fixture_report.json` and
`capture/graph_signatures.json` inside the release bundle.
