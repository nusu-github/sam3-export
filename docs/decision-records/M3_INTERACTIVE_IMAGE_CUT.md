# M3 — interactive image cache cut

## Decision

Ship `InteractiveImageEncodeInitial` as the fused image-only encode artifact and
retain `InteractiveFeatureProject` and `InitialNoMemoryCondition` as separate
logical component contracts. Do not publish the logical split as a fallback.

The deployment cut remains after the three conditioned image features because
their image-session lifetime spans repeated prompt predictions. Multimask
policy is represented by the two static `InteractivePredictMultimask3` and
`InteractivePredictSingle1` artifacts. No video memory or commit boundary is
part of this decision.

## Applicable profiles

- Plan `sam3_base_interactive_image_pvs_ortcuda_v1`.
- SAM3 base interactive image PVS with point, box and prior-mask prompts;
  exclusions are text image PCS, video/memory state, object batching and
  SAM3.1.
- Batch 1, image 1008, point capacity 16, box capacity 1, mask input 288,
  FP16: `b1-1008-p16-box1-mask288-fp16`.
- ONNX opset 18, ONNX Runtime 1.27.0 CUDA EP and CUDA IOBinding on NVIDIA RTX
  4000 Ada, driver 580.65.06.

Other shapes, precisions, backends and memory-aware image views require a new
decision.

## Evidence

The fused and logical-split eager recipes produced exact-equal image features
on the owned fixture. With two warmups and eight repeats, fused median/p95 was
189.419/189.566 ms and logical split was 189.600/189.811 ms. Both observed
1,336,356,352 peak PyTorch-allocated bytes and performed no intermediate D2H.
The small latency difference is recorded, not treated as a portable design
constant. Fusion is selected because the internal projection/conditioning
boundary has no independent lifetime, fan-out, host policy or backend
compatibility reason in this profile.

The three packaged graph pairs total 939,728,084 bytes:

| Artifact | ONNX + external data bytes |
|---|---:|
| `interactive-image-encode-initial` | 921,829,924 |
| `interactive-predict-multimask3` | 8,922,022 |
| `interactive-predict-single1` | 8,976,138 |

Official eager and local eager image features were exactly equal. Across the
zero-point, one-point, box, 16-point-plus-box, mask-only, mixed and repeated
click cases, official-to-local task-mask IoU was 0.990777–0.999959, score
maximum absolute difference was at most 0.001770, and low-resolution-logit
mean absolute difference was at most 0.038961. Top-score indices and output
mask counts matched.

The ORT CUDA/Public API gate retained all three image features as CUDA
`OrtValue`s. Its owned fixture executed one image encode and eight predictions
with nine session launches, 8,752,912 H2D bytes and 4,644,920 D2H bytes; D2H
was exactly the final scores and low-resolution logits. The repeated-click
trace used one image encode and two predictions and launched zero memory
encodes or commits. ORT median/p95 prediction latency was 8.909/91.372 ms. The
stage-boundary device-used samples observed 5,918,162,944 bytes at peak and a
3,514,826,752-byte increase from the pre-session sample; these device-level
observations are not portable capacity requirements.

## Replay

Fixture: `tests/fixtures/m3_interactive/cases.json`
(`m3-interactive-image-pvs-v1`), warmup 2, eight repeats, seed 20260726.
Official SAM3 commit: `cdff5a9`; checkpoint SHA-256:
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

```bash
PYTHONPATH=src uv run python scripts/export_interactive_image_v2.py \
  --official-repo ../sam3 --checkpoint /path/to/sam3.pt
```

The ignored release bundle owns the exact graph hashes, graph signatures,
four-stage parity, cache/copy counters and environment report.
