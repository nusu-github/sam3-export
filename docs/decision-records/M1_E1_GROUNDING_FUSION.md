# M1 E1 — grounding fusion

## Decision

Approve `GroundingFull` as the **M2 default candidate** for the applicable
profile below. Keep the corrected text-only encoder/decoder split as a
**fallback candidate** when backend compatibility requires a cut. Neither
candidate is shipped by this decision.

The shipped `sam3-split-onnx-v1` graphs remain a separate legacy plan. They
omit the official empty-geometry CLS prompt token and are not the parity
baseline or a future default.

## Applicable profiles

- SAM3 base, text-only image PCS; exclusions remain geometry/exemplar prompts,
  semantic output, interactive PVS, video and SAM3.1.
- Batch 1, image 1008, text length 32, 200 detector queries, FP16.
- ONNX Runtime 1.27.0 CUDA EP, CUDA `OrtValue` IOBinding, raw-200 output.
- NVIDIA RTX 4000 Ada (20,475 MiB), driver 580.65.06.

Other backends, shapes and output policies require their own record.

## Evidence

| Recipe | Median / p95 | Persistent / peak VRAM | Launches | D2H bytes | Artifact bytes |
|---|---:|---:|---:|---:|---:|
| Corrected split | 79.172 / 80.620 ms | 4,913 / 4,913 MiB | 2 | 33,179,602 | 70,063,978 |
| `GroundingFull` | 78.440 / 81.375 ms | 3,939 / 3,939 MiB | 1 | 33,179,602 | 72,235,758 |

Fusion removes one session launch and improves the measured median by
0.732 ms. Its p95 is 0.755 ms higher in this ten-sample run, so that difference
is not treated as a general performance constant. The observed VRAM values are
device `memory.used` above a pre-run baseline, sampled every 50 ms; they are
process-level arena observations, not portable capacity requirements.

Fused-versus-split parity preserved the 0.5 admitted-index set on all five
cases. Top-16 mask IoU ranged from 0.993750 to 0.999938, score max-absolute
difference from 0.0000031 to 0.0123615, and box max-absolute difference from
0.000488 to 0.002808.

The four-stage check was:

1. official eager -> corrected local eager: admitted-index sets matched on all
   cases; top-16 mask IoU was 0.982567–0.998427;
2. local eager -> `ExportedProgram`: static `torch.export(strict=False)` capture;
   encoder max-absolute difference was 0 and dense decoder max-absolute
   difference was 0.125 in FP16 mask logits;
3. local eager -> ORT fused: score max-absolute difference was
   0.0000084–0.0211546 and top-16 mask IoU was 0.988894–0.999580;
4. corrected split -> fused ORT: policy parity is reported above.

## Replay

Fixture: `tests/fixtures/m1_image_pcs/cases.json` (`m1-image-pcs-v1`), warmup 3,
10 repeats, seed 20260726. Candidate implementation: `22d0f27`; measurement
harness: `eb1d42c`; official SAM3: `cdff5a9`; checkpoint SHA-256:
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

```bash
PYTHONPATH=src uv run --no-sync python scripts/m1_experiments.py all \
  --work-dir .m1-work \
  --fixtures tests/fixtures/m1_image_pcs/cases.json \
  --official-root ../sam3 \
  --checkpoint /path/to/sam3.pt
```

Generated ONNX files and JSON reports stay in ignored `.m1-work/`; the fixture,
harness and this record are the replayable source of truth.
