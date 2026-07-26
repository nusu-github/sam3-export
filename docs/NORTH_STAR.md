# North star — exportable SAM3 (not “more Triton”)

## Product goal

**Make SAM3 usable as composable, `torch.export`-able inference subgraphs**,
unlike the official tree which optimizes for server PyTorch (`compile` /
multi-GPU / perflib), not portable export.

Downstream consumers may be:

- ExecuTorch
- ONNX (via export/dynamo paths)
- TensorRT / ORT / other AOT stacks that ingest `ExportedProgram` or equivalent

The **format is secondary**. The gate is always:

> Can this module (or this cut) be captured by **`torch.export`** with a
> stable tensor I/O contract?

Package name remains `sam3` for history; **the mission is no longer
“rewrite SAM3 in Triton.”**

## Non-goals

- Beating Meta on raw CUDA kernel microbenchmarks
- Shipping C++ / custom CUDA extensions / `torch.utils.cpp_extension` as
  required product dependencies
- One-shot export of the entire interactive + video + text runtime as a
  single giant graph (state machines and host control stay outside)

## Hard rules (dependency & implementation)

1. **Anything on the default inference path inside a model module must be
   `torch.export`-friendly** (ATen / standard `nn` / SDPA / ops with proper
   meta kernels). If it cannot export, it is not “internal default.”
2. **No required C++** for the product path. Optional research kernels must
   not be imported at module import time on the export path.
3. **Host-side / data-dependent control flow** (NMS loops, Hungarian
   association, variable-length Python lists, video state machines) lives in
   a **thin runtime outside** `ExportedProgram`, not inside exported
   subgraphs.
4. **Parity with official SAM3** remains for accuracy; **architecture for
   deploy** is ours (split I/O, fixed or explicitly dynamic shapes).
5. **New dependencies** need an export note: “used only outside graph” or
   “proven under `torch.export` on CUDA/CPU as applicable.”

### Allowed on export path (examples)

- `torch.nn.*`, `F.linear`, `F.layer_norm`, `F.scaled_dot_product_attention`
- Pure tensor ops, fixed-shape (or `Dim`-declared) masks/prompts
- Real-valued RoPE buffers (no `complex` dtype on export path)
- **Library blocks** that export cleanly: `timm.layers` primitives (DropPath,
  LayerScale, PatchEmbed construction), `torchvision.ops.box_convert`, stock
  `nn.MultiheadAttention` / `nn.Linear` — prefer these over hand-rolled clones
  when checkpoint load (or remap) still works

### Forbidden as default internals (examples)

- Hand Triton kernels (`@triton.jit`) are not in default production internals
- `scipy.optimize` inside a module `forward`
- `.item()` / `.tolist()` driven branching on the hot export path
- Custom C++ ops without a full export/meta story (we simply do not take this on)

### Optional / experimental (never default production)

- Hand Triton kernels are optional research paths only; not in the default build
  behind flags; must not break `export_mode` / ATen path
- `torch.compile` for server demos: allowed, orthogonal to export CI

## Decomposition (export cuts)

Prefer **small ExportedPrograms** composed by a Python (or app) runtime:

```
VisionEncoder     : image[B,3,H,W]     → multi-scale feats + PE
TextEncoder       : token ids         → text feats
PromptEncoder     : points/boxes/masks→ sparse + dense prompt
Det/Mask heads    : feats + prompts   → logits / boxes / low-res masks
Tracker step      : fixed-shape mem   → updated tokens / mask (when shape-stable)
```

Out of graph:

- tokenization string→ids (or pre-export tokens)
- score thresholding policy, NMS, det↔track association
- multi-object bookkeeping, frame loops, hotstart heuristics

## Acceptance criteria (project-level)

| Level | Criterion |
|-------|-----------|
| L0 | Documented I/O contract per subgraph |
| L1 | `torch.export` smoke green (fixed shapes) in CI |
| L2 | Re-run `ExportedProgram` matches eager within agreed atol/rtol |
| L3 | Optional dynamic dims where product needs them |
| L4 | One downstream sink demo (ExecuTorch **or** ONNX) for a single cut |

We climb L0→L4 **per subgraph**, not whole-SAM3 at once.

## Relation to past sprints

- Sprints 1–10: layer parity + product predictors (foundation we keep)
- Sprint 11: cede GEMM to cuBLAS; Triton left out of default product path
- **From here:** every new change answers *“does this help or hurt
  `torch.export`?”* first

See also: `docs/EXPORT_POLICY.md`.
