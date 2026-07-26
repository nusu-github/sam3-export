# Public deployment artifact catalog

This document is the short public tensor-I/O catalog for shipped deployment
artifacts. It is deliberately not a list of every module under `sam3.export`.
Logical components and test-only wrappers are cataloged separately in
[INTERNAL_COMPONENTS.md](INTERNAL_COMPONENTS.md); plan composition and backend
dispatch live in [DEPLOYMENT_PLANS.md](DEPLOYMENT_PLANS.md).

The terms in this document follow [GLOSSARY.md](GLOSSARY.md). In particular, a
logical component is not automatically a separately distributed artifact.

## Shipped deployment bundles

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
| `vision_encoder.onnx` | `pixel_values[1,3,1008,1008]` fp16 -> three detector FPN features and three positional tensors | Cache per normalized image. The legacy ABI exposes positional outputs that M1 will evaluate for pruning. |
| `text_encoder.onnx` | `input_ids[1,32]` int64, tokenizer-valid `attention_mask[1,32]` bool -> `text_memory[1,32,256]`, key-padding `text_padding_mask[1,32]` | Cache by text, tokenizer revision and model revision. |
| `grounding_encoder.onnx` | Low-resolution image feature/position/mask plus text memory/mask -> fusion memory and fixed level metadata | Legacy split boundary only. No host policy runs at this boundary. |
| `grounding_decoder.onnx` | Three image features plus grounding-encoder continuation -> 200 query logits, normalized `cxcywh` boxes, 200 mask logits and presence logits | Final tensors may cross to host for thresholding, selection, resize and rendering. |

The exact legacy tensor names, dtypes and shapes remain machine-readable in the
v1 manifest. The manifest does not describe a deployment plan, cache keys,
capture metadata, file hashes or external-data files; those are requirements
of the draft v2 contract rather than retroactive claims about v1.

## Planned artifacts are not shipped artifacts

M1 will compare a fused `GroundingFull`, vision-output pruning and an optional
selected-K recipe. Until the corresponding decision records are approved and
M2 release gates pass, those names appear only as proposed plans in
[DEPLOYMENT_PLANS.md](DEPLOYMENT_PLANS.md), never as shipped entries here.

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
