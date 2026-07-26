# sam3

`sam3` is an export-oriented SAM3 implementation. The package name is
historical: the supported inference path is standard PyTorch/ATen and is
designed to be captured as small `torch.export` graphs.

The project deliberately separates reusable tensor graphs from host-side
orchestration. Vision, text, prompt, and single-step tracking components are
the export-facing surface; tokenization, NMS, association, and frame loops stay
in the Python runtime.

See [the export policy](docs/EXPORT_POLICY.md) and
[the export-cut design](docs/EXPORT_CUTS.md) for contracts and boundaries.

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

## Export cuts

The current public cuts use tensor-only inputs and outputs:

- `VisionTower` / `VisionTowerFlat`: image → SAM3/SAM2 multi-scale features.
- `TextTower`: token ids and tokeniser attention mask → batch-first text memory.
- `PromptEncode`: fixed-size point prompts → sparse and dense embeddings.
- `InteractiveDecode`: image embeddings plus fixed-size points → masks and IoU.
- `InteractiveImageEmbed`: cached SAM2 FPN → image embedding plus high-res views.
- `GroundingEncode` / `GroundingDecode`: cached image/text tensors → fixed-query
  boxes, scores, and masks.
- `MemoryEncode` / `TrackerStep`: a predicted mask → tracker memory, and one
  fixed-shape tracker update. The runtime owns memory-bank selection and loops.

```python
from sam3.export import VisionTowerFlat
from sam3.weights import build_production_vision_backbone

neck = build_production_vision_backbone(load_weights=True, add_sam2_neck=True)
module = VisionTowerFlat(neck)
# exported = torch.export.export(module, (pixel_values,))
```

Run all fixed-shape CUDA export round trips with:

```bash
PYTHONPATH=src python scripts/export_smoke.py
```

Runtime postprocessing is not part of an exported graph:

```python
from sam3.runtime import associate_det_trk, nms_masks
```

## Installation

```bash
pip install -e '.[test]'
```

`timm`, `torchvision`, and TensorDict support the model components and tracker
state. `scipy` is used only by the host-side Hungarian association routine; it
is never imported by the export cuts.

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
