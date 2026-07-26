# Public export cuts

This document is the public tensor-I/O map for `sam3.export`. It describes
what each wrapper owns, what can be cached, and what stays in the host runtime.
Implementation details and model construction live in the Python modules named
below.

## Conventions

- One exported artifact has static shapes. `B=1` is the baseline; batching and
  dynamic dimensions are separate compatibility work.
- A boolean `*_mask` follows PyTorch key-padding convention unless noted:
  `True` means the token is ignored.
- Image preprocessing and conversion back to the original image size are host
  work. The production vision contract uses a fixed square image.
- `VisionTowerFlat` returns tensors in the order defined by
  `sam3.export.contracts.flat_vision_keys` so runtimes do not need Python
  dataclasses.

## Cut catalog

| Cut | Public module | Tensor contract | Cache / host boundary |
|---|---|---|---|
| Vision | `VisionTower`, `VisionTowerFlat` | `pixel_values[B,3,S,S]` → SAM3 FPN/PE and optional SAM2 FPN/PE levels | Cache per image or frame. Use `VisionTowerFlat` for an exported tensor tuple; `VisionTower` is the eager named output. |
| Text | `TextTower` | `input_ids[B,L]`, `attention_mask[B,L]` → `text_memory[B,L,D]`, `text_padding_mask[B,L]` | Cache per tokenized prompt. Its input mask is the tokenizer convention (`True` = valid); its output mask is key-padding convention. Tokenization is host work. |
| Interactive image view | `InteractiveImageEmbed` | Four cached SAM2 FPN levels → `image_embed`, `high_res_0`, `high_res_1` | Reuses VisionTower output. It applies the tracker mask-head high-resolution projections and initial-frame embedding; temporal scheduling is host work. |
| Point prompt | `PromptEncode` | Fixed `point_coords[B,N,2]`, `point_labels[B,N]` → sparse and dense prompt embeddings | The shipped wrapper is the fixed-point contract used by its tests. Point padding / UI policy is host work. |
| Interactive mask | `InteractiveDecode` | Fixed image embedding plus fixed point coordinates and labels → multimask logits and IoU | A self-contained tiny SAM-head export. It encodes its input points internally; consumers that need direct sparse/dense-prompt decoding must provide a separate wrapper. |
| Grounding encoder | `GroundingEncode` | Static tuples of image features, PEs, and image masks plus batch-first text memory/mask → memory, PE, padding mask, level metadata, text memory | Cache output while the image features and text prompt remain unchanged. |
| Grounding decoder | `GroundingDecode` | Image feature tuple plus `GroundingEncode` outputs → fixed-query logits, `cxcywh` boxes, masks, and presence scores | Thresholding, coordinate conversion, and NMS are host work. |
| Mask memory | `MemoryEncode` | Image features and a predicted mask → mask-memory features and positional encoding | Cache one result per selected tracker memory slot. The default wrapper treats masks as logits. |
| Tracker step | `TrackerStep` | Current image embedding/PE, high-res features, padded spatial memory, padded object-pointer memory, and fixed points → low/high-res mask logits, object token, object score | Runs one object on one frame. The host selects slots, constructs temporal positions, appends `MemoryEncode` output, and owns frame/object loops. |

## Composition

```
pixels ── VisionTowerFlat ──┬── SAM3 features ── GroundingEncode ── GroundingDecode
                            │                       ▲
token ids ── TextTower ────┘                       │
                                                    │
SAM2 features ── InteractiveImageEmbed ── TrackerStep ── MemoryEncode
```

`PromptEncode` and `InteractiveDecode` are independent interactive cuts. The
current `InteractiveDecode` is intentionally self-contained; it does not take
the output tuple of `PromptEncode`.

## Static-shape policy

| Value | Contract choice |
|---|---|
| Image side | Fixed for an artifact; the production vision contract is 1008×1008. |
| Text length | Fixed `L` with padding mask. |
| Point count | Fixed `N` with padding labels. |
| Detector queries | Fixed by the decoder weights. |
| Tracker spatial-memory slots | Fixed `M`; padded slots use the memory padding mask. |
| Tracker object-pointer slots | Fixed `P`; padded slots use the object padding mask. |

## Runtime checklist

1. Preprocess pixels and tokenize text outside exported programs.
2. Cache VisionTower and TextTower outputs under application-owned keys.
3. Invoke only the downstream cut affected by a prompt, object, or frame
   change.
4. Apply selection, association, cache policy, and display-space conversion in
   the host runtime.

Run `PYTHONPATH=src python scripts/export_smoke.py` to verify eager/export
round trips for the shipped public cuts.
