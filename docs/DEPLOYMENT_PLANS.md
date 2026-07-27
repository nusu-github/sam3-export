# Deployment plans

This document owns public plan composition, representative traces, backend
dispatch and release status. Low-level public tensor I/O belongs in
[EXPORT_CUTS.md](EXPORT_CUTS.md), and component details belong in
[INTERNAL_COMPONENTS.md](INTERNAL_COMPONENTS.md).

Lifecycle and dispatch role are separate. **Shipped**, **candidate** and
**planned** describe availability/maturity. **Default**, **optional** and
**fallback** describe dispatch. A selected-K recipe can therefore be a
candidate with an intended optional role without calling it shipped.

## M5 shipped SAM3.1 Multiplex video plan

`sam3_1_multiplex_video_tracking_ortcuda_v1` is shipped/default only for
**SAM3.1 multiplex video tracking / point-box-mask correction / bucket16 / ORT
CUDA v1**. It is packaged under
`artifacts/sam3-multiplex-video-tracking-ortcuda-v2/` with the fixed
`fixed-bucket1-dispatch1to2-1008-p16-mask288-m10-ptr16-fp16` profile.

```text
MultiplexFrameEncode -- Tri per-frame CUDA cache -------------------+
                                                                   |
interaction: host resolves object -> bucket/slot                   |
             MultiplexInteractionPreview (non-mutating)            |
                 -> MultiplexScatterReplaceCommit -- selected slot |
                                                                   |
propagation: MultiplexPropagation fixed B1 <-----------------------+
                 -> one dispatch for bucket 0
                 -> second independent dispatch only for bucket 1
```

The Host Runtime owns public object IDs, stable lowest-free-slot assignment
and assignment revision. It dispatches the same fixed B1 artifact once for
1–16 objects and twice for 17–32 objects. Normal frame progression retains
shared memory, validity and pointers in each bucket's CUDA state; demux occurs
only when ordering final public results by object ID.

Selected interaction is preview-only until a single-mask handle is committed.
Commit scatter-replaces the target slot once, and non-target slot/bucket state
remains byte-identical. CUDA EP and IOBinding are required, D2H is final-public
output only, and there is no fallback. `MultiplexStateV1` is not M4
`BaseVideoStateV1`.

The Public API/state contract is
[MULTIPLEX_VIDEO_API.md](MULTIPLEX_VIDEO_API.md). The measured B1 dispatch
adoption and bounded-dynamic rejection are recorded in
[the M5 decision record](decision-records/M5_SAM31_MULTIPLEX_PROFILE.md).

## M4 shipped base-video plan

`sam3_base_video_tracking_ortcuda_v1` is shipped/default only for **SAM3 base
video tracking / point-box-mask correction / per-object batch / ORT CUDA v1**.
It is packaged under `artifacts/sam3-base-video-tracking-ortcuda-v2/` with the
fixed `b4-1008-p16-box1-mask288-m10-ptr16-fp16` profile. M2 image PCS and M3
interactive image remain defaults in their own use cases.

```text
TrackerFrameEncode -- per-frame CUDA cache ---------------------+
                                                                |
correction: BaseTrackerPreviewMultimask3 (display only)         |
                    -> BaseTrackerPreviewSingle1                |
                    -> BaseMemoryCommit                         |
                                                                |
propagation: BaseTrackerStepAndCommitSingle1 <------------------+
                    -> host-owned BaseVideoStateV1
```

Correction preview is non-mutating and only the final single1 handle may be
committed. Steady-state propagation uses the measured fused artifact. The
measured B4 profile packs independent objects with validity padding: 1 and 4
objects use one tracker launch, while 5 use two; every frame encode runs once.
No memory or attention is shared across objects.

The frame pyramid, memory, pointer and commit tensors remain CUDA-resident.
Only public scores, final low-resolution logits and final masks cross D2H.
CUDA EP and IOBinding are required; there is no implicit CPU, M3 image-cache or
legacy fallback. The state/cache and Public API contract is
[BASE_VIDEO_API.md](BASE_VIDEO_API.md), and B4/fused adoption is recorded in
[the M4 decision record](decision-records/M4_BASE_VIDEO_CUT.md).

## M3 shipped interactive image plan

`sam3_base_interactive_image_pvs_ortcuda_v1` is shipped/default only for
**SAM3 base interactive image PVS / point-box-mask / ORT CUDA v1**. It is
packaged under `artifacts/sam3-interactive-image-pvs-ortcuda-v2/` with the
fixed `b1-1008-p16-box1-mask288-fp16` profile. It does not replace the M2
text-only image PCS default.

```text
InteractiveImageEncodeInitial
  = VisionTrunk + SAM2Neck + InteractiveFeatureProject
    + InitialNoMemoryCondition
                  |
                  +-- image-session CUDA cache --+
                                                   |
              host multimask policy ---------------+
                  |                                |
                  v                                v
 InteractivePredictMultimask3        InteractivePredictSingle1
```

The image-only recipe fuses feature projection and the checkpoint-owned
initial/no-memory condition after keeping their logical component contracts
separate. The cache key includes that condition and the static profile, so it
cannot represent an M4 memory-aware image view. Multimask is host dispatch
between two static artifacts, not a runtime tensor branch: three masks are the
Public API default, while repeated click can feed one selected low-resolution
logit to the single-mask artifact.

The three cached image tensors remain CUDA-resident through every prediction.
Only scores and final low-resolution logits cross D2H. CUDA EP and IOBinding
are required; there is no CPU pyramid, legacy split or text-PCS fallback. The
plan has no video/object/memory state and performs no preview, memory encode or
commit. Those transitions are owned by the separate M4 plan.

The fused image-only cut and its applicable profile are fixed by the
[M3 decision record](decision-records/M3_INTERACTIVE_IMAGE_CUT.md).

## M2 shipped plans

All three plans use SAM3 base text-only image PCS, B1/1008/L32/Q200/FP16,
ONNX Runtime 1.27.0 CUDA EP and CUDA `OrtValue` IOBinding. They are packaged
together under `artifacts/sam3-image-pcs-ortcuda-v2/`; the generated manifests
are the runtime-readable source of truth.

| Plan ID | Lifecycle / dispatch | Composition | Output policy / fallback |
|---|---|---|---|
| `sam3_base_image_pcs_text_ortcuda_v1` | shipped / default | required-position-only `DetectorImageEncode` + `TextEncoder` + fused `GroundingFull` | raw-200 final output |
| `sam3_base_image_pcs_text_ortcuda_selected_k32_v1` | shipped / optional | image/text cache + corrected grounding encoder + query core + K32 mask continuation | fixed K=32; fallback is the fused raw-200 default |
| `sam3_base_image_pcs_text_ortcuda_split_v1` | shipped / fallback | image/text cache + corrected grounding encoder/decoder | raw-200 compatibility cut |

The optional plan copies only compact scores/boxes/presence to host before
selection and uploads fixed `selected_indices[1,32]` plus
`valid_mask[1,32]`; continuation tensors remain CUDA-resident. With zero
proposals the mask graph is skipped. Failure to provide CUDA EP or IOBinding
is a capability error, not an implicit CPU or legacy dispatch.

## Legacy shipped plan

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

## M1 decision history

The M1 decisions below established the M2 input. The applicable candidates are
now shipped only through the three M2 manifests above.

### Fused image PCS candidate

```text
DetectorImageEncode = VisionTrunk + DetectorNeck
TextEncoder
GroundingFull        = Fusion + DETR + Mask
```

The default plan ID is `sam3_base_image_pcs_text_ortcuda_v1`. It preserves
independent image and text cache lifetimes while removing the legacy grounding
encoder/decoder boundary. M1 E1 approves `GroundingFull` as its default
recipe and the corrected text-only split as a compatibility fallback. The
legacy v1 split is neither of those candidates. See
[the E1 record](decision-records/M1_E1_GROUNDING_FUSION.md).

M1 E2 approves the required-position-only vision boundary as the default
candidate. Feature-only is optional when a profile explicitly prioritizes the
smaller boundary, and full output is compatibility fallback only. See
[the E2 record](decision-records/M1_E2_VISION_OUTPUTS.md).

### Policy / selected-K optional candidate

```text
GroundingQueryCore -> compact scores/boxes/presence -> host selection
        |                                            |
        +-- device continuation state ---------------+
                                                     v
                                      GroundingMaskSelectedK
```

M1 E3 approved fixed K=32 as the shipped optional M2 plan. Selection occurs only
after the final all-query interaction, and `valid_mask` preserves a fixed
shape. K=16 and K=64 were measured but are not admitted to the initial plan.
A public split recipe requires device-resident continuation handoff. A backend
that cannot provide it dispatches to fused raw-200 `GroundingFull`, never by
copying large continuation tensors through CPU/NumPy. See
[the E3 record](decision-records/M1_E3_SELECTED_K.md).

## Later plans

Additional backend bundles (M6) remain planned. M4 `BaseVideoStateV1` and M5
`MultiplexStateV1` remain separate semantic ABIs.

## Decision and publication gates

- E1, E2 and E3 decision records contain replayable fixtures, median/p95
  latency, peak/persistent VRAM, copy bytes, session count, artifact bytes and
  four-stage parity.
- A plan manifest identifies default, optional and fallback applicability per
  backend/profile. An unrecorded fallback is not permitted.
- The M2 manifests, M3 interactive manifest, M4 base-video manifest and M5
  Multiplex manifest passed schema/package validation and their CUDA/Public
  API release validators; later plan changes require the same gate.
- The Host Runtime dispatches by public plan ID and resolved manifest; it does
  not expose backend tensor names in the Public API.
