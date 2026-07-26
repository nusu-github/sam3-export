# M0 artifact and documentation inventory

Status: reviewed M0 baseline at `sam3-export` commit
`847db18e0558816d88d592f24a45c36593ae1e8e` and official-model repository
commit `cdff5a927beba49b6249c1e2973b29bd64f40b83`.

This remains the historical M0 inventory. M2 additions are owned by
[EXPORT_CUTS.md](EXPORT_CUTS.md) and [DEPLOYMENT_PLANS.md](DEPLOYMENT_PLANS.md);
they do not retroactively add metadata to the legacy files below.

## Shipped package inventory

| Item | Classification | Scope / disposition |
|---|---|---|
| `artifacts/sam3-onnx/manifest.json` | Legacy shipped manifest | Format `sam3-split-onnx-v1`; remains unchanged and is dispatched separately from v2 |
| `vision_encoder.onnx` | Legacy split stage | SAM3 detector image features; M1 E2 will evaluate excess outputs |
| `text_encoder.onnx` | Legacy split/cache stage | SAM3 text-only prompt encoding |
| `grounding_encoder.onnx` | Legacy split stage / M1 baseline | No host policy at its output boundary |
| `grounding_decoder.onnx` | Legacy split stage / M1 baseline | Raw 200-query image PCS result |
| Four `.onnx.data` files | Legacy external data | Required package files but absent from v1 manifest integrity metadata |
| Tokenizer files | Legacy runtime data | Used by the Space; v1 does not inventory or hash them |
| `spaces/sam3-onnx/app.py` | Legacy example Host Runtime | Hard-coded v1 filenames, shapes and positional wiring; downloads but does not parse the v1 manifest |

Together these items implement **SAM3 text-only image PCS / legacy split
v1**. They do not implement geometry/exemplar prompts, semantic output,
production interactive PVS, video or SAM3.1.

## Internal source inventory

| Class | Members | M0 disposition |
|---|---|---|
| Components backing legacy graphs | `VisionTowerFlat`, `TextTower`, `GroundingEncode`, `GroundingDecode` | Keep component tests and legacy generation; do not declare the grounding cut a future default |
| Other internal/prototype components | `VisionTower`, `DetectorEncoderDecoder`, `InteractiveImageEmbed`, `MemoryEncode`, `TrackerStep` | Keep in internal catalog; milestone-specific production contracts remain pending |
| Tiny test-only fixtures | Default `PromptEncode`, default `InteractiveDecode`, synthetic grounding/tracker builders in tests | Label test-only and exclude from the public artifact catalog |
| Diagnostic output profile | `VisionTowerFlat(add_sam2=True)` | Export/parity fixture, not a universal multi-capability ABI |

## v1 gaps that must not be guessed during v2 migration

The v1 manifest contains graph filenames and tensor names/dtypes/shapes plus
global image size, text length and opset. It does not contain:

- plan ID, lifecycle/status, full scope or exclusions;
- source repository commit and checkpoint digest;
- semantic tensor types, coordinates, units, normalization and validity;
- cache lifetime/key/invalidation and state compatibility;
- backend/device handoff requirements or fallback dispatch;
- capture mode, range constraints, fixture hashes or parity records;
- graph, external-data, tokenizer and report sizes/hashes; or
- baked versus Host Runtime policy.

The Space additionally hard-codes graph filenames, dimensions and positional
output indexes. M2 added a separate manifest-driven bundle/runtime; this M0
inventory records the legacy limitation without changing its behavior.

## Documentation disposition

| Surface | M0 source of truth |
|---|---|
| Terms and scope labels | `GLOSSARY.md` |
| Public artifact tensor catalog | `EXPORT_CUTS.md` |
| Internal component/fixture classification | `INTERNAL_COMPONENTS.md` |
| Plan composition/status/dispatch | `DEPLOYMENT_PLANS.md` |
| Machine-readable v2 contract | `schemas/sam3-deployment-manifest-v2.schema.json` |
| v2 field meaning, ownership and migration | `MANIFEST_V2.md` |

README and Space text are summaries of these sources. Regenerating the legacy
model card through `scripts/export_onnx.py` must preserve the exact legacy
scope label.
