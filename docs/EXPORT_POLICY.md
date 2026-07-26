# Export policy

`sam3.export` contains the deployable inference boundaries. A boundary is
accepted only when it can be captured and re-executed with `torch.export` for
its documented static input shapes.

```python
from torch.export import export

program = export(module.eval(), args, strict=False)
actual = program.module()(*args)
```

## Requirements for an exported cut

| Concern | Requirement |
|---|---|
| Inputs and outputs | Tensors, or a fixed tuple of tensors, only. |
| Shapes | Image side, sequence length, point count, query count, and memory-slot count are static for one artifact. |
| Operations | Standard PyTorch / ATen operations with an export path. |
| State | All state needed by `forward` is a registered parameter or buffer. |
| Validation | Eager output and `ExportedProgram.module()` output match within the test tolerance. |
| Coverage | The cut is included in `scripts/export_smoke.py`. |

Dynamic dimensions may be added to a cut only after its static contract and
round-trip test are green.

## Keep outside the graph

The following are runtime responsibilities, not exported graph operations:

- string tokenization and BPE;
- image/video decoding, resizing, and original-image coordinate conversion;
- score thresholds, NMS, connected components, and mask selection;
- detection-to-track association and other combinatorial matching;
- frame loops, object banks, cache eviction, and temporal-slot selection.

The runtime may compose exported cuts freely, but it must pass tensors that
match the selected artifact's fixed contract.

## Dependency rule

An import on a model path must be export-compatible. Packages used only by the
runtime must remain outside `sam3.export` and module `forward` methods. In
particular, an exported cut must not require custom CUDA/C++ extensions,
Triton kernels, NumPy/SciPy algorithms, or Python values derived from tensor
data to decide its execution path.

## Verification

Run the fixed-shape CUDA round-trip suite after changing any cut or a primitive
used by a cut:

```bash
PYTHONPATH=src python scripts/export_smoke.py
```

Use the corresponding test file for faster iteration. The smoke suite covers
the public wrappers; ordinary unit tests still cover the underlying model
components and host runtime.
