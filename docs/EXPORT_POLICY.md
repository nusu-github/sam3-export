# Export policy — `torch.export` gate

## Gate API (conceptual)

```python
import torch
from torch.export import export

ep = export(module, args, kwargs, dynamic_shapes=..., strict=False)
# must re-execute:
out = ep.module()(*args, **kwargs)
```

If a candidate **model-internal** dependency or layer cannot pass this for
the shapes we care about, it does not land on the default path.

## Modes

| Mode | Meaning |
|------|---------|
| **export / production default** | ATen-only style: `F.linear`, SDPA, `nn.LayerNorm` (or equivalent export-safe LN), real RoPE |
| **runtime experiments** | Optional kernel experiments (Triton/bnb/custom CUDA) are not in the default path |

No backend-switch env var is required for default export paths.

## Subgraph checklist (before claiming “exportable”)

- [ ] Pure `nn.Module.forward`, tensor in / tensor (or tuple of tensors) out
- [ ] No `scipy` / numpy host algorithms inside `forward`
- [ ] No required unregistered Triton or custom CUDA in model internals
- [ ] No complex dtype on the path
- [ ] Control flow is shape-static or uses `torch.cond` / explicit masks
- [ ] Documented example `args` + optional `dynamic_shapes`
- [ ] Parity test: eager vs `ep.module()` within tolerances
- [ ] Listed in `scripts/export_smoke.py`

## Runtime outside the graph

These may use Python freely (still prefer no C++):

- BPE / string tokenization
- `nms_masks`, connected components (until registered as export custom ops —
  default: run after export)
- `associate_det_trk` (Hungarian)
- Video `init_state` / `propagate` orchestration
- Multi-prompt product policy

## Dependency review template

When adding a package or heavy import:

```
Name:
Used in:  [export subgraph | runtime only | optional-research]
torch.export status:  [proven | blocked | n/a]
Fallback if blocked:
```

## Smoke

```bash
cd /opt/sam3/sam3
PYTHONPATH=src python scripts/export_smoke.py
# or
PYTHONPATH=src pytest tests/test_export_smoke.py -q
```
