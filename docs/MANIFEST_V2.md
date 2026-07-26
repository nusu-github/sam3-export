# Deployment manifest v2 release contract

The JSON Schema at
[`schemas/sam3-deployment-manifest-v2.schema.json`](../schemas/sam3-deployment-manifest-v2.schema.json)
defines the machine-readable contract for a deployment **plan bundle**, not an
individual logical component. It uses JSON Schema Draft 2020-12 and fixes
`format` to `sam3-deployment-manifest-v2`.

M2, M3 and M4 use this schema for fixed image PCS, interactive image and base
video ORT CUDA plans. The current `sam3-split-onnx-v1` manifest remains a
separate legacy format. Loaders dispatch on `format`; they must not treat v1 as
a partial v2 document or invent missing provenance, policy, hash, cache or
handoff values.

## Required blocks

| Block | Contract |
|---|---|
| `scope` | Public/internal/fixture/legacy classification, lifecycle, separate dispatch role, use case, capabilities and explicit exclusions |
| `plan` | Stable plan ID, semantic graph kind, role set, contract version and composed canonical components |
| `model` | Family/variant/layout, source repository and commit, model revision, checkpoint SHA-256 and variant-owned parameters |
| `backend` / `profile` | Target/backend/EP, relevant versions and opset, capability flags, precision, static constants and bounded dimensions |
| `tensors` | Backend-independent semantic tensor definitions including shape, layout, coordinates/value kind, normalization, padding, validity and residency |
| `artifacts` / `execution` | Backend files and tensor bindings plus plan edges; backend tensor names remain bindings rather than semantic type names |
| `caches` | Cached tensors, lifetime, versioned key parts, invalidation and state compatibility |
| `handoffs` | Producer/consumer edge, tensor set, required/preferred/host-allowed residency mechanism and explicit fallback |
| `capture` | Canonical capture kind/mode, PyTorch/exporter versions, constraints and graph-signature file |
| `policies` | Explicit baked or Host Runtime ownership for selection, output, multimask, branch and related policy |
| `fixtures` | Versioned owner, source/checkpoint, case inventory, aggregate hash and four-stage parity results |
| `files` | Relative path, role, byte size and SHA-256 for every graph, external-data, tokenizer, capture and report file |

The manifest itself is excluded from its own file inventory. If a package
digest is added later, it is computed over sorted `(path, size_bytes, sha256)`
records so it has no self-hash cycle.

## Ownership

| Concern | Accountable owner |
|---|---|
| Vocabulary, scope and exclusions | Architecture Owner |
| Schema version, package population and file/external-data hashes | Release/Backend Owner |
| Semantic tensor, cache, state and handoff contracts | Runtime Lead with the relevant component owner |
| Official reference fixture contents, source revision and aggregate hash | Export Tech Lead |
| Public API plan ID and contract version | API Owner + Runtime Lead |
| Experiment measurements and adoption decision | Export Tech Lead + Runtime Lead; Product/Integration Owner for optional selected-K |

The schema fixture under `tests/fixtures/manifest_v2/` is a minimal skeleton of
the M2 fused default plan. It demonstrates structural validation only; the
generated M2/M3/M4 bundle manifests and owned fixtures satisfy their release
gates.

## Validation layers

JSON Schema validates the release blocks, main types, current vocabulary and
hash form. The common package validator additionally checks unique IDs,
artifact/file/tensor references, file existence/size/SHA-256, required CUDA
IOBinding capabilities and named fallback-plan presence. M2, M3 and M4 adapters
then check scope and declared input/output bindings against each ORT graph. It
does not implement a general DAG executor or future-backend cross-rule matrix.

Generated manifests live at `manifests/<plan-id>.json` in ignored release
bundles. Their format is `sam3-deployment-manifest-v2` and contract version is
`1.0.0`. Current profiles are `b1-1008-l32-q200-fp16` for image PCS,
`b1-1008-p16-box1-mask288-fp16` for interactive image PVS, and
`b4-1008-p16-box1-mask288-m10-ptr16-fp16` for base video. v1 remains
separately dispatched and receives no inferred v2 metadata.
