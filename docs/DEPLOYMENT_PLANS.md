# Deployment plans

This document owns public plan composition, representative traces, backend
dispatch and release status. Low-level public tensor I/O belongs in
[EXPORT_CUTS.md](EXPORT_CUTS.md), and component details belong in
[INTERNAL_COMPONENTS.md](INTERNAL_COMPONENTS.md).

Lifecycle and dispatch role are separate. **Shipped**, **candidate** and
**planned** describe availability/maturity. **Default**, **optional** and
**fallback** describe dispatch. A selected-K recipe can therefore be a
candidate with an intended optional role without calling it shipped.

## Shipped plan

### SAM3 text-only image PCS / legacy split v1

| Property | Value |
|---|---|
| Classification | Legacy shipped artifact |
| Lifecycle | Shipped |
| Dispatch role | Legacy; not the future default |
| Package format | `sam3-split-onnx-v1` |
| Representative trace | One normalized image and one tokenized text prompt produce 200 boxes/scores/mask logits; host selects at most 25 for the demo |
| Composition | `vision_encoder` + `text_encoder` -> `grounding_encoder` -> `grounding_decoder` |
| Backend | ONNX Runtime CUDA EP with CUDA `OrtValue` IOBinding |
| Device handoff | Required for the documented GPU path between all four sessions |
| Fallback | No manifest-driven fallback in v1; the Space wiring is fixed |
| Scope exclusions | Geometry/exemplar prompts, semantic output, production interactive PVS, video and SAM3.1 |

The package remains usable within this scope. M0 does not rewrite its manifest
or graphs and does not infer missing provenance, cache or file-integrity
metadata.

## M1 candidates

No candidate in this section is shipped or approved as a default at M0.

### Fused image PCS candidate

```text
DetectorImageEncode = VisionTrunk + DetectorNeck
TextEncoder
GroundingFull        = Fusion + DETR + Mask
```

The intended plan ID is `sam3_base_image_pcs_text_ortcuda_v1`. It preserves
independent image and text cache lifetimes while removing the legacy grounding
encoder/decoder boundary. M1 E1 must compare fused and split graphs with the
same 200-mask output policy, fixtures, dtype, warmup, IOBinding and device
residency. M1 E2 separately decides which vision outputs remain public.

If E1 rejects the fused recipe for an applicable backend/profile, the approved
split recipe remains the fallback under a distinct manifest and explicit
status; backend differences do not change the Public API.

### Policy / selected-K optional candidate

```text
GroundingQueryCore -> compact scores/boxes/presence -> host selection
        |                                            |
        +-- device continuation state ---------------+
                                                     v
                                      GroundingMaskSelectedK
```

This candidate is optional and pending M1 E3. Selection occurs only after the
final all-query interaction. `K` is a fixed profile (initial comparison:
16/32/64 and, if useful, 25) with a `valid_mask`, not a data-dependent output
axis. A public split recipe requires device-resident continuation handoff. A
backend that cannot provide it must dispatch to fused `GroundingFull` or a
graph-internal top-k recipe rather than copy large continuation tensors through
CPU/NumPy.

## Later plans

Interactive image PVS (M3), SAM3 base video (M4), SAM3.1 native Multiplex
(M5) and additional backend bundles (M6) remain planned. Their architectural
names do not reserve public artifact IDs or contracts at M0. In particular,
SAM3 base object batching and SAM3.1 16-slot bucket-space Multiplex require
separate state ABIs.

## Decision and publication gates

- E1, E2 and E3 each receive a decision record with replayable fixtures,
  median/p95 latency, peak/persistent VRAM, copy bytes, session count,
  artifact bytes and four-stage parity.
- A plan manifest identifies default, optional and fallback applicability per
  backend/profile. An unrecorded fallback is not permitted.
- Candidate names move into the shipped catalog only after the M2 release
  contract and manifest validation pass.
- The Host Runtime dispatches by public plan ID and resolved manifest; it does
  not expose backend tensor names in the Public API.
