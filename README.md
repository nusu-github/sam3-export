# sam3

`sam3` is an export-oriented SAM3 implementation. The package name is
historical: its tensor components use standard PyTorch/ATen and are designed
for `torch.export` capture and composition into deployment plans.

The project separates Public API, Host Runtime, canonical tensor components and
deployment plans/artifacts. Python wrappers in `sam3.export` are an internal
component surface, not a promise of one distributed graph per wrapper.
Tokenization, NMS, association, cache policy and frame loops stay in the Host
Runtime.

See the [glossary](docs/GLOSSARY.md), [component policy](docs/EXPORT_POLICY.md),
[public artifact catalog](docs/EXPORT_CUTS.md), and
[deployment plans](docs/DEPLOYMENT_PLANS.md). The M2 Public API is specified in
[IMAGE_PCS_API.md](docs/IMAGE_PCS_API.md).

## Package layout

```text
sam3/
  primitives/  # ATen-only attention, MLP, RoPE, positional encodings
  vision/      # ViTDet, FPN necks, prompt encoder, interactive mask head
  grounding/   # text tower and image-text detection/segmentation components
  tracking/    # memory encoder, tracker step, and video orchestration
  runtime/     # host-only NMS, association, mask geometry, video input
  export/      # tensor-only torch.export wrappers and I/O contracts
  weights/     # checkpoint loading and production builders
```

Import from the package that owns the concern. The former catch-all
`sam3.layers` namespace is intentionally gone.

```python
from sam3.export import VisionTower, VisionTowerFlat
from sam3.vision import PromptEncoder, SamImageHead
from sam3.grounding import VETextEncoder
from sam3.tracking import Sam3Tracker
from sam3.runtime import nms_masks
```

## Internal export components

The current Python wrappers use tensor-only inputs and outputs. They include
components and test-only fixtures; consult the public artifact catalog before
treating any wrapper as a supported deployment boundary.

- `VisionTower` / `VisionTowerFlat`: image → SAM3/SAM2 multi-scale features.
- `TextTower`: token ids and tokeniser attention mask → batch-first text memory.
- `PromptEncode`: fixed-size tiny point fixture → sparse and dense embeddings.
- `InteractiveDecode`: self-contained tiny interactive fixture → masks and IoU.
- `InteractiveImageEmbed`: cached SAM2 FPN → image embedding plus high-res views.
- `GroundingEncode` / `GroundingDecode`: legacy split components and M1
  fused-vs-split baseline → fixed-query boxes, scores, and masks.
- `MemoryEncode` / `TrackerStep`: a predicted mask → tracker memory, and one
  fixed-shape tracker update. The runtime owns memory-bank selection and loops.

```python
from sam3.export import VisionTowerFlat
from sam3.weights import build_production_vision_backbone

neck = build_production_vision_backbone(load_weights=True, add_sam2_neck=True)
module = VisionTowerFlat(neck)
# exported = torch.export.export(module, (pixel_values,))
```

Run the fixed-shape CUDA component/fixture round trips with:

```bash
PYTHONPATH=src python scripts/export_smoke.py
```

Runtime postprocessing is not part of an exported graph:

```python
from sam3.runtime import associate_det_trk, nms_masks
```

M2 ships **SAM3 base text-only image PCS / ORT CUDA v1** for the fixed
B1/1008/L32/Q200/FP16 profile. Its manifest-driven plans are fused raw-200
default, selected-K32 optional and corrected split fallback. Build the ignored
release bundle with:

```bash
PYTHONPATH=src python scripts/export_image_pcs_v2.py \
  --official-repo ../sam3 --checkpoint /path/to/sam3.pt
```

The previous **SAM3 text-only image PCS / legacy split v1** bundle remains
usable and separately dispatched; it is not the new default or fallback.
Neither bundle covers geometry/exemplar prompts, production interactive PVS,
video, semantic output or SAM3.1.

## Installation

```bash
pip install -e '.[test]'
```

`timm`, `torchvision`, and TensorDict support the model components and tracker
state. `scipy` is used only by the host-side Hungarian association routine; it
is never imported by the export cuts.

The core package includes manifest validation but delays importing ONNX
Runtime until session creation. Install the pinned CUDA runtime only when using
the M2 bundle:

```bash
pip install -e '.[ort-cuda]'
```

## Quality checks

The development toolchain is pinned in `pyproject.toml` and is managed with
[`uv`](https://docs.astral.sh/uv/). Install it and run the complete local gate
with:

```bash
uv sync --all-groups --inexact
make quality
```

For iterative work, `make format` applies safe Ruff fixes and formatting;
`make lint`, `make typecheck`, `make test`, and `make build` run the individual
checks. Pull requests and pushes to `main` run the same `make quality` gate in
GitHub Actions.

## License

This repository redistributes and adapts SAM3 materials under the
[SAM License](LICENSE).
