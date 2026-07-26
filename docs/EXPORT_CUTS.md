# Export cuts — where to slice for reuse

Goal: **high reuse** of expensive compute and of exported artifacts, while
keeping orchestration (NMS, association, video loops) outside the graph.
The public package is `sam3`; **these cut names** should survive as its public
surface.

Design principle:

> Cut where **cache lifetime** and **modality** change — not where a Python
> class happens to sit today.

---

## 0. Layers of reuse

```
┌─────────────────────────────────────────────────────────────┐
│  L3 Product runtime (Python, not exported)                  │
│  interactive session · text-on-video · multi-object policy  │
└───────────────┬─────────────────────────────┬───────────────┘
                │ calls                       │ calls
┌───────────────▼──────────────┐  ┌───────────▼────────────────┐
│  L2 Task graphs (export OK)  │  │  L2 Task graphs            │
│  InteractiveDecode           │  │  GroundingDecode           │
│  TrackerStep                 │  │  MemoryEncode              │
└───────────────┬──────────────┘  └───────────┬────────────────┘
                │ consumes cached             │
┌───────────────▼─────────────────────────────▼────────────────┐
│  L1 Feature towers (export OK, heavy, cacheable)             │
│  VisionTower (sam3 FPN + sam2 FPN) · TextTower · (optional)  │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│  L0 Primitives (export OK, tiny)                             │
│  PromptEncode · RoPE · LN · Linear · SDPA blocks             │
└──────────────────────────────────────────────────────────────┘
```

- **L1** = maximum **compute reuse** (one image encode → many prompts / frames).
- **L2** = maximum **product remix** (same feats, different heads).
- **L3** = maximum **UX flexibility** (must stay out of `torch.export`).

If a cut does not sit cleanly in L1 or L2 with **tensor-only I/O**, it is not
an export unit.

---

## 1. Recommended cuts (canonical)

### Cut A — `VisionTower`

| | |
|--|--|
| **What** | Dual ViT-Det neck: image → sam3 FPN + sam2 FPN (+ PE) |
| **In** | `pixel_values: [B,3,S,S]` fixed `S` (e.g. 1008) |
| **Out** | Named tuple / dict of tensors only: |
| | `sam3_fpn: list[[B,C_i,H_i,W_i]]` or packed levels |
| | `sam3_pe:  same structure` |
| | `sam2_fpn: list[...]` |
| | `sam2_pe:  ...` |
| **Implementation status** | **landed** — `sam3.export.contracts` + `VisionTower` / `VisionTowerFlat`; tests in `tests/test_vision_tower_export.py` |
| **Reuse** | Interactive, open-vocab det, video every frame, shared cache |
| **Export** | Highest priority. Pure conv/transformer, fixed spatial size. |
| **Not inside** | PIL resize, letterbox policy (preproc is L3 or separate tiny util) |

VisionTower contract path: `sam3.export.vision_tower` (`VisionTower`, `VisionTowerFlat`).

Official already almost has this: `SAM3VLBackbone.forward_image` /
`VisionOnly`. **This is the spine** — every other graph hangs off its outputs.

**Why this cut wins reuse:**  
Text prompts, clicks, and tracks all share the same pixels. One exported
`VisionTower` serves three products.

---

### Cut B — `TextTower`

| | |
|--|--|
| **What** | Token ids (+ optional box placeholders) → text memory |
| **In** | `input_ids: [B,L]`, `attention_mask: [B,L]` (tokens **precomputed**) |
| **Out** | `text_memory: [B,L,D]`, `text_mask: [B,L]` (bool/float) |
| **Reuse** | Open-vocab image, text-on-video seed, future multi-prompt batch |
| **Export** | High priority. Keep **string BPE outside** (string → ids = L3). |
| **Official** | `forward_text` but strip `List[str]` from the exported module |
| **Implementation status** | **landed** — `sam3.export.TextTower`; input masks use tokeniser convention (`True` = valid), output masks use PyTorch key-padding convention (`True` = padded). Tests in `tests/test_remaining_export_cuts.py`. |

**Cut rule:** never put `List[str]` or Python tokenizer inside ExportedProgram.
Export starts at **ids**.

---

### Cut C — `InteractiveImageEmbed` (optional thin fuse of A)

| | |
|--|--|
| **What** | VisionTower sam2 branch + scalp + `conv_s0/s1` + `no_mem_embed` |
| **In** | same as A, or **consumes** cached A outputs |
| **Out** | `image_embed: [B,C,h,w]`, `high_res_0`, `high_res_1` |
| **Reuse** | SAM-click / box interactive; also tracker “image side” |
| **Why separate from A** | Det path wants **sam3** FPN; interactive wants **sam2** + no_mem. Same tower, different **view**. |

Prefer: export **A**, and a tiny **C'** that is only:

```
(sam2_fpn levels) → (image_embed, high_res*)
```

so det and interactive share one VisionTower artifact.

**Implementation status:** **landed** — `sam3.export.InteractiveImageEmbed`
consumes the four flat SAM2 FPN tensors, applies the tracker head's high-res
projections, and adds the initial-frame `no_mem_embed`. The runtime keeps frame
memory scheduling outside this view.

---

### Cut D — `PromptEncode`

| | |
|--|--|
| **What** | points / boxes / masklets → sparse + dense prompt embeddings |
| **In** | fixed max points `N_max` (pad + mask), optional box, optional mask |
| **Out** | `sparse: [B,N,C]`, `dense: [B,C,H,W]` |
| **Reuse** | Interactive decode, iterative refine, tracker clicks |
| **Export** | **Landed** — `sam3.export.PromptEncode` (fixed `N=2` points → sparse `N+1` with pad); tests in `tests/test_prompt_interactive_export.py` |

---

### Cut E — `InteractiveDecode` (SAM head)

| | |
|--|--|
| **What** | image_embed + high_res + prompt embeds → masks / iou / low_res |
| **In** | outputs of C + D |
| **Out** | `low_res_logits [B,K,h,w]`, `iou [B,K]`, (optional upsampled masks) |
| **Reuse** | Click demo, refine-after-text, tracker mask head |
| **Export** | **Landed (tiny head)** — `sam3.export.InteractiveDecode` wraps `SamImageHead` with fixed points + multimask; production 256-d wiring is a later scale-up |

Upsample to original H×W can stay L3 (dynamic original size).

---

### Cut F — `GroundingEncode` (fusion encoder)

| | |
|--|--|
| **What** | sam3 multi-scale feats + text memory → fused encoder memory |
| **In** | subset of A outs + B outs |
| **Out** | flat `memory [seq,B,C]`, `spatial_shapes`, masks, text-after-enc |
| **Reuse** | All open-vocab decode variants; multi-prompt if text batched |
| **Export** | Medium difficulty (DETR-style multi-scale). Worth it: **det body**. |
| **Implementation status** | **landed** — `sam3.export.GroundingEncode`; static tuple FPN inputs and tensor-only flattened outputs. Tests in `tests/test_remaining_export_cuts.py`. |

---

### Cut G — `GroundingDecode` (decoder + heads)

| | |
|--|--|
| **What** | encoder memory → boxes, mask logits, presence / class scores |
| **In** | F outs (+ queries) |
| **Out** | fixed `Q` queries: `pred_boxes [B,Q,4]`, `pred_logits [B,Q,*]`, `pred_masks [B,Q,h,w]` |
| **Reuse** | Image text det; per-frame det in video |
| **Export** | Keep **Q fixed**. Threshold / NMS = L3. |
| **Implementation status** | **landed** — `sam3.export.GroundingDecode`; decoder, score head, and segmentation head stay weight-sharing wrappers. Tests in `tests/test_remaining_export_cuts.py`. |

---

### Cut H — `MemoryEncode` (mask → memory feature)

| | |
|--|--|
| **What** | SimpleMaskEncoder / CX stack |
| **In** | image feats + binary/soft mask |
| **Out** | `maskmem [B,C_m,h,w]` (e.g. C_m=64) |
| **Reuse** | Every tracker step after a mask exists |
| **Export** | Small, high leverage for video packaging |
| **Implementation status** | **landed** — `sam3.export.MemoryEncode`; accepts logits (or soft masks by static configuration) and returns features plus positional encoding. Tests in `tests/test_remaining_export_cuts.py`. |

---

### Cut I — `TrackerStep` (one step, fixed shapes)

| | |
|--|--|
| **What** | mem-attn + SAM head for **one** object, **one** frame feature pack |
| **In** | current frame vision pack + stacked memory tokens (padded `M_max`) + prompt/obj tokens |
| **Out** | updated object tokens, mask logits, (optional) new memory to append |
| **Reuse** | Pure track, text-seeded track, multi-object via **batch dim or outer loop** |
| **Export** | Hardest L2; still the right cut. **Frame loop and object bank = L3**. |

**Implementation status:** **landed** — `sam3.export.TrackerStep` consumes a
fixed, padded spatial-memory bank plus fixed object-pointer slots and emits one
object's masks, score, and object token. `MemoryEncode` is intentionally a
separate artifact for the host to append a new slot. Slot selection and temporal
position construction remain L3. The tracker RoPE attention now accepts a
standard key-padding mask so padded slots are genuinely inert.

Do **not** export `propagate()` or `init_state()`. Export **one step**.

---

## 2. What must NOT be a cut (L3 runtime)

| Keep outside export | Why |
|---------------------|-----|
| String tokenizer / BPE | Host strings |
| `nms_masks`, score threshold | Data-dependent keep set |
| `associate_det_trk` / Hungarian | scipy / combinatorial |
| `Sam3TextOnVideo.propagate` | Object×frame state machine |
| MultiGPU gather | Distributed |
| PIL / video decode | I/O |

These compose L1/L2 artifacts; they are the **product differentiator** relative
to a frozen mega-ONNX, without fighting export.

---

## 3. Reuse matrix (why these cuts)

| Consumer \ Producer | Vision A | Text B | Prompt D | Inter E | Ground F+G | Mem H | Track I |
|---------------------|----------|--------|----------|---------|------------|-------|---------|
| Click interactive   | ✓ (sam2 view) | | ✓ | ✓ | | | |
| Text open-vocab     | ✓ (sam3) | ✓ | | | ✓ | | |
| Video track (points)| ✓ per frame | | ✓ | ✓ | | ✓ | ✓ |
| Text-on-video       | ✓ | ✓ | | | ✓ seed | ✓ | ✓ |
| Refine text w/ click| ✓ cache | ✓ once | ✓ | ✓ | optional | | |

**Single VisionTower export** is the highest ROI cut: four rows share it.

---

## 4. Caching & lifetime (product UX)

```
set_image / encode_frame:
    VisionTower(pixels) → cache key = image_id or frame_idx
    optional InteractiveImageEmbed view for click path

set_text:
    TextTower(ids) → cache by prompt hash

predict_click / decode_grounding / track_step:
    only L2 graphs; read caches; no re-encode
```

Reuse rule of thumb:

- **Pixels change rarely** → cut A must be independently invocable and cacheable.
- **Prompts change often** → D/E/G must not re-run A.
- **Time advances** → I runs per frame; A may run per frame or use a short window cache.

---

## 5. Shape policy (export-friendly)

| Tensor | Policy |
|--------|--------|
| Image side | Fixed `S×S` (1008) inside graph; orig H×W only in L3 |
| Text length | Fixed `L_max` + mask (pad) |
| Points | Fixed `N_max` + padding mask |
| Det queries | Fixed `Q` |
| Memory bank | Fixed `M_max` slots for TrackerStep |
| Batch | `B` dynamic later; start static `B=1` |

Dynamic dims only after static export is green.

---

## 6. Public package layout

Suggested layout (illustrative):

```
sam3/
  export/                 # or sam3/deploy/
    contracts.py          # Typed dicts / dataclass I/O
    vision_tower.py       # thin nn.Module wrappers, ATen-only
    text_tower.py
    interactive.py
    grounding.py
    tracker_step.py
    runtime/              # L3: sessions, NMS, associate, video
  # existing model/* remains training + full Python reference
```

Packaging principles:

1. **Wrappers over existing modules** first (re-export friendly forwards), not a second weight tree.
2. Official keeps `compile` path; export path is `forward_export_*` or dedicated modules that call the same weights.
3. Drop lab Triton backends at merge (or isolate under `sam3.lab`) — default is ATen.
4. Contracts in one file so mobile / ORT / ExecuTorch all speak the same names.

---

## 7. Priority order (implementation)

| Order | Cut | Why |
|------:|-----|-----|
| 1 | **A VisionTower** | Shared by everything; heaviest; already dual-neck shaped |
| 2 | **D PromptEncode** | Already export-green; unblocks interactive split |
| 3 | **C view + E InteractiveDecode** | End-to-end click without L3 model class |
| 4 | **B TextTower** | ids-in only; enables det without strings in graph |
| 5 | **F+G Grounding** | Open-vocab package |
| 6 | **H MemoryEncode** | Video building block |
| 7 | **I TrackerStep** | Full video story; hardest |

Do **not** start with whole `Sam3TextOnVideo` — it is L3 by nature.

---

## 8. Anti-patterns (low reuse)

| Bad cut | Problem |
|---------|---------|
| Whole `SamInteractivePredictor` including PIL | Can't export; couples I/O |
| `forward(image, List[str])` VL backbone | Strings + vision fused → can't cache text/vision independently |
| Per-object full model export | No multi-object batch; duplicates VisionTower |
| NMS inside decoder graph | Dynamic keep indices break static consumers |
| One mega-graph “all modalities” | No product remix; every app pays full size |

---

## 9. One-sentence summary

**Cut at feature towers (vision / text) and at single-shot decoders (prompt→mask, fusion→det, memory, one tracker step); leave sessions, strings, and matching outside.**  
That maximizes reuse of exports and of runtime cache, and maps cleanly onto a future non-triton `sam3` deploy package.
