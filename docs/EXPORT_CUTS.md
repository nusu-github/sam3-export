# Public deployment artifact catalog

This document is the short public tensor-I/O catalog for shipped deployment
artifacts. It is deliberately not a list of every module under `sam3.export`.
Logical components and test-only wrappers are cataloged separately in
[INTERNAL_COMPONENTS.md](INTERNAL_COMPONENTS.md); plan composition and backend
dispatch live in [DEPLOYMENT_PLANS.md](DEPLOYMENT_PLANS.md).

The terms in this document follow [GLOSSARY.md](GLOSSARY.md). In particular, a
logical component is not automatically a separately distributed artifact.

## Shipped deployment bundles

### SAM3.1 multiplex video tracking / point-box-mask correction / bucket16 / ORT CUDA v1

| Property | Contract |
|---|---|
| Status | Shipped v2 bundle; default only within the SAM3.1 Multiplex video use case |
| Backend/profile | ORT 1.27.0 CUDA EP + IOBinding; `fixed-bucket1-dispatch1to2-1008-p16-mask288-m10-ptr16-fp16` |
| Package manifest | `sam3_1_multiplex_video_tracking_ortcuda_v1`, contract `1.0.0` |
| Host boundary | RGB frame/prompt validation, public object lifecycle and assignment, fixed B1 dispatch 1–2 times, final object-ID ordering and mask conversion |
| Exclusions | SAM3 base, text/geometry PCS, streaming, CPU fallback, unbounded dynamic buckets and M6 backend bundles |

| Public artifact role | Short tensor I/O / boundary |
|---|---|
| `multiplex-frame-encode` | normalized `pixel_values[1,3,1008,1008]` fp16 -> Tri interactive and propagation 72/144/288 frame views; one CUDA cache entry per frame |
| `multiplex-interaction-preview-multimask3` | selected slot's interactive view + P16/box1/mask288 prompt validity -> three display masks/scores and private commit candidates; non-mutating |
| `multiplex-interaction-preview-single1` | same selected-slot prompt path -> one mask/score and private commit mask/pointer/score |
| `multiplex-propagation-bucket1` | propagation frame view + one fixed `[16,...]` `MultiplexStateV1` bucket -> native multiplex mask/score/pointer updates; dispatched once or twice |
| `multiplex-memory-commit-bucket1` | selected single-mask commit value -> conditioning shared-memory feature/position for one bucket |
| `multiplex-scatter-replace-commit-bucket1` | selected slot result + existing bucket state -> scatter-replaced slot and committed memory while non-target slots remain unchanged |

Exact bucket tensors, variant parameters, graph signatures, hashes and device
handoffs are manifest-owned. The host-visible state/lifecycle contract is
[MULTIPLEX_VIDEO_API.md](MULTIPLEX_VIDEO_API.md).

### SAM3 base video tracking / point-box-mask correction / per-object batch / ORT CUDA v1

| Property | Contract |
|---|---|
| Status | Shipped v2 bundle; default only within the SAM3 base-video use case |
| Backend/profile | ORT 1.27.0 CUDA EP + IOBinding; `b4-1008-p16-box1-mask288-m10-ptr16-fp16` |
| Package manifest | `sam3_base_video_tracking_ortcuda_v1`, contract `1.0.0` |
| Host boundary | RGB frame validation/packing, object/revision/slot lifecycle, frame cache, fixed B4 chunking, final mask conversion |
| Exclusions | M2 image PCS and M3 interactive-image dispatch, streaming video, CPU fallback, SAM3.1 Tri neck/bucket state/Multiplex |

| Public artifact role | Short tensor I/O / boundary |
|---|---|
| `tracker-frame-encode` | normalized `pixel_values[1,3,1008,1008]` fp16 -> unconditioned `image_embedding/image_position[1,256,72,72]`, `high_res_0[1,32,288,288]`, `high_res_1[1,64,144,144]`; one CUDA frame cache entry |
| `base-tracker-preview-multimask3` | cached frame view + B4 fixed `BaseVideoStateV1` + P16/box1/mask288 prompt validity -> three low-resolution masks/scores and device commit candidates; preview-only |
| `base-tracker-preview-single1` | same inputs -> one low-resolution mask/score, object pointer/score and CUDA commit mask; only this preview can be committed |
| `base-memory-commit` | cached frame view + final single device outputs -> one conditioning memory feature/position per valid object; correction commit only |
| `base-tracker-step-and-commit-single1` | cached frame view + B4 state + empty propagation prompt -> single prediction, pointer/score and non-conditioning memory in one steady-state launch |

Exact B4 state tensors, backend names, hashes and residency are manifest-owned.
The host-visible contract is [BASE_VIDEO_API.md](BASE_VIDEO_API.md).

### SAM3 base interactive image PVS / point-box-mask / ORT CUDA v1

| Property | Contract |
|---|---|
| Status | Shipped v2 bundle; default only within the interactive image PVS use case |
| Backend/profile | ORT 1.27.0 CUDA EP + IOBinding; `b1-1008-p16-box1-mask288-fp16` |
| Package manifest | `sam3_base_interactive_image_pvs_ortcuda_v1`, contract `1.0.0` |
| Host boundary | Image/prompt validation and packing, original-to-model coordinates, image cache lifecycle, multimask dispatch, final resize/threshold |
| Exclusions | Text image PCS dispatch, video/memory state, object batching and SAM3.1 |

| Public artifact role | Short tensor I/O / boundary |
|---|---|
| `interactive-image-encode-initial` | normalized `pixel_values[1,3,1008,1008]` fp16 -> `image_embedding[1,256,72,72]`, `high_res_0[1,32,288,288]`, `high_res_1[1,64,144,144]`; image-session CUDA cache fixed to the initial/no-memory condition |
| `interactive-predict-multimask3` | cached CUDA features + P16/box1/mask288 fixed prompt/validity tensors -> float32 `low_res_logits[1,3,288,288]` and `scores[1,3]` |
| `interactive-predict-single1` | same fixed prompt ABI -> float32 `low_res_logits[1,1,288,288]` and `scores[1,1]`; official single-mask stability policy is baked |

Exact bindings, prompt validity, cache key, file hashes and required device
handoffs are manifest-owned. The Public API contract is
[INTERACTIVE_IMAGE_API.md](INTERACTIVE_IMAGE_API.md).

### SAM3 base text-only image PCS / ORT CUDA v1

| Property | Contract |
|---|---|
| Status | Shipped v2 bundle; fused default, selected-K32 optional, corrected split fallback |
| Model/capability | SAM3 base, text-only image PCS |
| Backend/profile | ORT 1.27.0 CUDA EP + IOBinding; B1/1008/L32/Q200/FP16 |
| Package manifests | `manifests/<plan-id>.json`, format `sam3-deployment-manifest-v2`, contract `1.0.0` |
| Host boundary | RGB preprocess/tokenize, cache lifecycle, strict threshold/stable selection, optional mask NMS, output conversion |
| Exclusions | See [the deployment plan](DEPLOYMENT_PLANS.md); interactive/video/SAM3.1 are not implied |

| Public artifact role | Short tensor I/O / boundary |
|---|---|
| `detector-image-encode` | normalized fp16 pixels -> three detector FPN features plus the required low-resolution position tensor; image-session CUDA cache |
| `text-encode` | token IDs + valid-token mask -> text memory + padding mask; prompt-session CUDA cache |
| `grounding-full` | cached image/text tensors + compact image mask -> raw 200 scores/boxes/mask logits/presence at the final host boundary |
| `grounding-encode` + `grounding-decode` | corrected text-only compatibility split used only by the explicit fallback plan |
| `grounding-query-core` + `grounding-mask-selected-k32` | compact host policy after all-query interaction; large continuation remains device-resident |

Exact backend names, shapes, cache keys, handoffs, policies and file hashes are
owned by the generated manifests rather than duplicated here. Corresponding
logical components remain cataloged in [INTERNAL_COMPONENTS.md](INTERNAL_COMPONENTS.md).

### SAM3 text-only image PCS / legacy split v1

| Property | Contract |
|---|---|
| Status | Shipped legacy bundle; supported for its documented scope, but not the default recipe for manifest v2 |
| Model family | SAM3 base (`facebook/sam3`) |
| Capability | Text-only image promptable concept segmentation (PCS) |
| Excluded capabilities | Geometry/exemplar prompts, semantic output, production interactive PVS, video tracking, SAM3.1 Tri neck and Multiplex |
| Backend/profile | ONNX Runtime CUDA EP with IOBinding; batch 1, fp16, 1008x1008 image, text length 32 |
| Package manifest | `artifacts/sam3-onnx/manifest.json`, format `sam3-split-onnx-v1` |
| Host boundary | Decode/resize, tokenization, thresholding, selection/NMS, output resize and rendering |
| Device handoff | Intermediate outputs must remain CUDA `OrtValue`s for the documented GPU path |

The bundle contains four graph files. They are legacy split stages, not four
independent product capabilities.

| File | Tensor contract | Cache / host boundary |
|---|---|---|
| `vision_encoder.onnx` | `pixel_values[1,3,1008,1008]` fp16 -> three detector FPN features and three positional tensors | Cache per normalized image. M1 found these positional outputs excessive for the v2 default, but the legacy ABI remains unchanged. |
| `text_encoder.onnx` | `input_ids[1,32]` int64, tokenizer-valid `attention_mask[1,32]` bool -> `text_memory[1,32,256]`, key-padding `text_padding_mask[1,32]` | Cache by text, tokenizer revision and model revision. |
| `grounding_encoder.onnx` | Low-resolution image feature/position/mask plus text memory/mask -> fusion memory and fixed level metadata | Legacy split boundary only. No host policy runs at this boundary. |
| `grounding_decoder.onnx` | Three image features plus grounding-encoder continuation -> 200 query logits, normalized `cxcywh` boxes, 200 mask logits and presence logits | Final tensors may cross to host for thresholding, selection, resize and rendering. |

The exact legacy tensor names, dtypes and shapes remain machine-readable in the
v1 manifest. The manifest does not describe a deployment plan, cache keys,
capture metadata, file hashes or external-data files; those are requirements
of the draft v2 contract rather than retroactive claims about v1.

## Later artifacts are not shipped artifacts

Additional backend/profile variants remain later milestones.
Internal smoke wrappers do not make them shipped artifacts.

## Catalog admission rule

A new entry is added only when all of the following are true:

1. its deployment plan and applicable backend/profile are approved;
2. its public semantic tensor contract and Host Runtime boundary are stable;
3. its plan manifest validates against the release schema;
4. official eager -> local eager -> `ExportedProgram` -> backend parity and
   end-to-end behavior are recorded against owned fixtures;
5. artifact and external-data hashes are available; and
6. default, optional or fallback status is explicit.

Internal export smoke coverage alone does not admit a component to this
catalog.
