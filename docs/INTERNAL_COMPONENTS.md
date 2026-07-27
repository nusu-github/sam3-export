# Internal export components and fixtures

This catalog classifies tensor wrappers used for structure, composition and
local parity. Inclusion here does not mean a standalone ONNX file is shipped.
Public deployment artifacts are listed only in
[EXPORT_CUTS.md](EXPORT_CUTS.md).

## Current component inventory

| Component / wrapper | Classification | Current role and limitation | Verification owner |
|---|---|---|---|
| `VisionTower` | Internal component | Named eager detector/SAM2 feature view. Not itself a stable backend tuple ABI. | Vision/export component tests |
| `VisionTowerFlat` | Internal component and legacy-stage producer | Flat tensor tuple used by export. `add_sam2=True` is diagnostic/multi-branch output, not a universal public ABI. | Vision export tests and legacy ONNX comparison |
| `VisionTowerProfiled` | Internal component used by M2 | Emits full, required-position-only or feature-only fixed tuples. M2 ships required-position-only for its fixed profile; wrapper existence alone does not admit other profiles. | M1 E2 + M2 release validator |
| `TextTower` | Internal component and legacy-stage producer | Independent text cache boundary with tokenizer-valid input mask and key-padding output mask. | Text/export component tests |
| `GroundingEncode` | Internal component and legacy split stage | Useful local parity cut. M1 approves only the corrected text-only split as an M2 compatibility fallback; legacy v1 remains distinct. | Grounding component tests and M1 E1 |
| `TextOnlyPromptEncode` / `GroundingEncodeTextOnly` | Internal components used by M2 | Corrected text-only path includes the official image-conditioned empty-geometry CLS token. It does not rewrite legacy v1. | M1 official/local parity + M2 release validator |
| `GroundingDecode` | Internal component and legacy split stage | Produces dense 200-query image PCS outputs. Public standalone status is not implied. | Grounding component tests and M1 E1/E3 |
| `GroundingFull` | Internal component used by M2 default | Fused grounding is packaged through the M2 default plan; the Python wrapper itself is not independently shipped. | M1 E1 + M2 release validator |
| `GroundingFullFeatureOnly` | M1 candidate component | E2 consumer that recomputes the required position tensor for the optional feature-only boundary. | M1 E2 harness and decision record |
| `GroundingQueryCore` / `GroundingMaskSelectedK` | Internal components used by M2 optional | E3 policy cut after final all-query interaction. M2 admits only K=32; large continuation remains device-resident. | M1 E3 + M2 release validator |
| `DetectorEncoderDecoder` | Internal component | Overlaps grounding responsibilities and remains an internal compatibility/test surface pending responsibility cleanup. | Detector component tests |
| `InteractiveImageEmbed` | Internal compatibility component | Older mixed projection/initial-condition view retained for component tests; it is not used by the M3 release plan. | Interactive export tests |
| `InteractiveFeatureProject` | M3 logical component | Projects production SAM2 FPN levels to the 72 base and 288/144 high-resolution image views without temporal conditioning. | M3 local/official parity and cut measurement |
| `InitialNoMemoryCondition` | M3 logical component | Adds the checkpoint-owned initial/no-memory condition to the 72x72 base view. Its identity is part of the M3 cache key. | M3 cache/repeated-click gate |
| `InteractiveImageEncodeInitial` | M3 plan component | Fuses vision, feature projection and initial conditioning for the shipped image-only artifact; this wrapper is not a claim that every logical component ships separately. | M3 export/CUDA validator |
| `InteractivePredictMultimask3` / `InteractivePredictSingle1` | M3 plan components | Production prompt encoder and mask decoder with fixed P16/box1/mask288 validity ABI and static output policy. | M3 official-to-Public parity |
| `TrackerFrameEncode` | M4 plan component | Produces the unconditioned 72x72 tracker view and 288/144 high-resolution views once per frame; the cache condition is distinct from M3. | M4 official-to-Public trajectory parity |
| `BaseTrackerPreviewMultimask3` / `BaseTrackerPreviewSingle1` | M4 plan components | Memory-aware B4 prediction with fixed `BaseVideoStateV1` and the M3 prompt ABI. Multimask is preview-only; single emits private device values required for commit. | M4 correction and trajectory gates |
| `BaseMemoryCommit` | M4 plan component | Converts the final single preview into conditioning memory. It is a correction boundary, not the steady-state default by itself. | M4 correction/replacement gates |
| `BaseTrackerStepAndCommitSingle1` | M4 plan component | Fused single preview and non-conditioning memory encode for steady-state propagation; selected only for the shipped B4 profile. | M4 fused-cut decision and CUDA validator |
| `Mux` / `Demux` | M5 canonical components | Convert host-owned assignment/validity between object order and fixed `[bucket,16,...]` tensors. Demux is used at the public result boundary, not for frame-to-frame large state. | M5 identity/padding/removal isolation tests |
| `MultiplexFrameEncode` | M5 plan component | Produces checkpoint-mapped Tri interactive and propagation views once per frame. It is distinct from the M4 dual/SAM2-neck frame view. | M5 official/local and CUDA frame parity |
| `MultiplexInteractionPreviewMultimask3` / `MultiplexInteractionPreviewSingle1` | M5 plan components | Run point/box/mask interaction for a host-selected slot. Preview is non-mutating; only single1 emits private scatter/commit candidates. | M5 selected-interaction and lifecycle gates |
| `MultiplexPropagation` | M5 canonical/plan component | Applies native shared-memory 16-slot propagation to one fixed bucket. The public two-bucket recipe dispatches this component twice rather than treating wrappers as one-to-one artifacts. | M5 1/2/15/16 and 17/32 trajectory gates |
| `MultiplexMemoryCommit` | M5 plan component | Encodes the selected single-mask result into checkpoint-native shared bucket memory. It does not reuse M4 memory scale/bias or per-object memory layout. | M5 correction/memory parity |
| `MultiplexScatterReplaceCommit` | M5 canonical/plan component | Replaces one selected slot and commits its conditioning memory while preserving all non-target slots/buckets. | M5 byte-isolation and stale-preview gates |
| `sam3.export.fixtures.PromptEncode` | Test-only fixture | Fixed tiny/default point contract used for unit export coverage. It is not the production prompt encoder artifact. | Prompt interactive export tests |
| `sam3.export.fixtures.InteractiveDecode` | Test-only fixture | Self-contained tiny SAM head; it does not consume `PromptEncode` outputs and is not a production pipeline. | Prompt interactive export tests |
| `MemoryEncode` | Internal component / fixture | Earlier generic-shape fixture retained for component tests. M4 production uses checkpoint-validated `BaseMemoryCommit`; this fixture is not its artifact ABI. | Remaining-cut regression tests |
| `TrackerStep` | Internal SAM3-base component / fixture | Earlier fixed-step fixture retained for component tests. M4 production artifacts are separately mapped; this remains neither a public plan nor SAM3.1 Multiplex. | Remaining-cut regression tests |

`scripts/export_smoke.py` verifies internal eager/`torch.export` round trips. Its
coverage is a component-quality gate, not a public artifact release list.

## Planned canonical components

Names such as `VisionTrunk` and `DetectorNeck` remain architectural component
names unless a milestone implements and verifies them. Listing a name does not
claim it is a separately distributed artifact.

Shape similarity does not establish semantic compatibility. In particular,
concept prompts and instance prompts, detector and interactive pyramids, and
single-object and mux16 memory tensors keep distinct semantic types and ABI
versions.

## Promotion path

An internal component may be used in a public plan only after the plan-level
composition, host boundary, backend/profile, capture constraints, reference
fixtures and release manifest are reviewed. Promotion normally fuses several
components; it does not rename every wrapper into an artifact.
