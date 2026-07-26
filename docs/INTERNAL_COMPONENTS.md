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
| `VisionTowerProfiled` | M1 candidate component | Emits full, required-position-only or feature-only fixed tuples for E2. The E2 record, not this wrapper, owns plan admission. | M1 E2 harness and decision record |
| `TextTower` | Internal component and legacy-stage producer | Independent text cache boundary with tokenizer-valid input mask and key-padding output mask. | Text/export component tests |
| `GroundingEncode` | Internal component and legacy split stage | Useful local parity cut. M1 approves only the corrected text-only split as an M2 compatibility fallback; legacy v1 remains distinct. | Grounding component tests and M1 E1 |
| `TextOnlyPromptEncode` / `GroundingEncodeTextOnly` | M1 candidate components | Corrected text-only path includes the official image-conditioned empty-geometry CLS token. It does not rewrite legacy v1. | M1 official/local parity harness |
| `GroundingDecode` | Internal component and legacy split stage | Produces dense 200-query image PCS outputs. Public standalone status is not implied. | Grounding component tests and M1 E1/E3 |
| `GroundingFull` | M1 candidate component | E1 fused grounding candidate approved as M2 default input for the recorded ORT CUDA profile; not a shipped artifact. | M1 E1 harness and decision record |
| `GroundingFullFeatureOnly` | M1 candidate component | E2 consumer that recomputes the required position tensor for the optional feature-only boundary. | M1 E2 harness and decision record |
| `GroundingQueryCore` / `GroundingMaskSelectedK` | M1 candidate components | E3 policy cut after final all-query interaction. Fixed K and `valid_mask`; large continuation remains device-resident. | M1 E3 harness and decision record |
| `DetectorEncoderDecoder` | Internal component | Overlaps grounding responsibilities and remains an internal compatibility/test surface pending responsibility cleanup. | Detector component tests |
| `InteractiveImageEmbed` | Internal component | Mixes image feature projection with initial/no-memory conditioning; M3 will separate the logical contracts. | Interactive export tests |
| `PromptEncode` | Test-only fixture | Fixed tiny/default point contract used for unit export coverage. It is not the production prompt encoder artifact. | Prompt interactive export tests |
| `InteractiveDecode` | Test-only fixture | Self-contained tiny SAM head; it does not consume `PromptEncode` outputs and is not a production pipeline. | Prompt interactive export tests |
| `MemoryEncode` | Internal component / fixture | Logical memory operation. Variant-specific scale/bias and single/mux layout are not a generic production ABI. | Remaining-cut tests; M4/M5 mapping tests later |
| `TrackerStep` | Internal SAM3-base component / fixture | Fixed per-object tracker step. It is neither a one-object-per-session mandate nor a SAM3.1 Multiplex implementation. | Remaining-cut tests; M4 trajectory tests later |

`scripts/export_smoke.py` verifies internal eager/`torch.export` round trips. Its
coverage is a component-quality gate, not a public artifact release list.

## Planned canonical components

Names such as `VisionTrunk`, `DetectorNeck`, `InteractiveFeatureProject`,
`InitialNoMemoryCondition`, `BaseTrackerPreview`, `MemoryEncodeSingle`, `Mux`,
`Demux`, `TrackPropagateMux16` and `MemoryEncodeMux16` are architectural
component names. They become code and fixtures in the milestone that needs
them; listing them does not claim they are implemented or shipped.

Shape similarity does not establish semantic compatibility. In particular,
concept prompts and instance prompts, detector and interactive pyramids, and
single-object and mux16 memory tensors keep distinct semantic types and ABI
versions.

## Promotion path

An internal component may be used in a public plan only after the plan-level
composition, host boundary, backend/profile, capture constraints, reference
fixtures and release manifest are reviewed. Promotion normally fuses several
components; it does not rename every wrapper into an artifact.
