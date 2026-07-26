# Export component policy

`sam3.export` contains canonical tensor components and export fixtures. It is
not the public deployment artifact catalog: exportability is necessary but is
not evidence that a wrapper should be packaged as a separate graph.

Architecture terms, scope labels and catalog classes are fixed in
[GLOSSARY.md](GLOSSARY.md). Shipped artifacts are admitted through
[EXPORT_CUTS.md](EXPORT_CUTS.md), and component composition is defined in
[DEPLOYMENT_PLANS.md](DEPLOYMENT_PLANS.md).

A logical component is accepted into the internal export surface only when it
can be captured and re-executed with `torch.export` for its documented input
profile.

```python
from torch.export import export

program = export(module.eval(), args, strict=False)
actual = program.module()(*args)
```

## Requirements for an internal export component

| Concern | Requirement |
|---|---|
| Inputs and outputs | Tensors, or a fixed tuple of tensors, only. |
| Shapes | Image side, sequence length, point count, query count, and memory-slot count are static for one artifact. |
| Operations | Standard PyTorch / ATen operations with an export path. |
| State | All state needed by `forward` is a registered parameter or buffer. |
| Validation | Eager output and `ExportedProgram.module()` output match within the test tolerance. |
| Coverage | The component or fixture is included in `scripts/export_smoke.py`. |

Dynamic dimensions may be added to a cut only after its static contract and
round-trip test are green.

## Keep outside the graph

The following are runtime responsibilities, not exported graph operations:

- string tokenization and BPE;
- image/video decoding, resizing, and original-image coordinate conversion;
- score thresholds, NMS, connected components, and mask selection;
- detection-to-track association and other combinatorial matching;
- frame loops, object banks, cache eviction, and temporal-slot selection.

The runtime may compose approved deployment artifacts, but it must pass
tensors that match the resolved plan manifest. Internal component contracts do
not independently authorize a runtime composition.

## Deployment-cut admission

A separately packaged cut must be justified by at least one of lifetime reuse,
fan-out, a compact Host Runtime policy decision, or backend compatibility. The
review also accounts for boundary tensor bytes, VRAM liveness, launches,
synchronization, lost fusion, duplicated parameters and ABI maintenance.

An internal smoke pass does not satisfy this gate. Public admission additionally
requires plan-level official/eager/export/backend/end-to-end parity, owned
fixtures, a validated manifest, file hashes and explicit default/optional/
fallback status.

## Dependency rule

An import on a model path must be export-compatible. Packages used only by the
runtime must remain outside `sam3.export` and module `forward` methods. In
particular, an exported cut must not require custom CUDA/C++ extensions,
Triton kernels, NumPy/SciPy algorithms, or Python values derived from tensor
data to decide its execution path.

## Verification

Run the fixed-shape CUDA round-trip suite after changing any component or a
primitive used by one:

```bash
PYTHONPATH=src python scripts/export_smoke.py
```

Use the corresponding test file for faster iteration. The smoke suite covers
internal component fixtures; ordinary unit tests still cover the underlying
model components and Host Runtime. Release coverage is tracked at plan level.
