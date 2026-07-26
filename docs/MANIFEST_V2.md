# Deployment manifest v2 draft

The draft JSON Schema at
[`schemas/sam3-deployment-manifest-v2.schema.json`](../schemas/sam3-deployment-manifest-v2.schema.json)
defines the machine-readable contract for a deployment **plan bundle**, not an
individual logical component. It uses JSON Schema Draft 2020-12 and fixes
`format` to `sam3-deployment-manifest-v2`.

This is an M0 draft. The current `sam3-split-onnx-v1` manifest remains a
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

The schema fixture under `tests/fixtures/manifest_v2/` is synthetic and owned
by the schema tests. It demonstrates structural validation; it is not an
official model parity fixture and cannot satisfy a release gate.

## Validation layers

JSON Schema validates required fields, vocabulary, hash syntax, the presence of
bounded-shape range fields and object closure. A later semantic/package linter
must additionally validate:

- ID uniqueness and every cross-reference;
- `minimum <= optimum <= maximum` for every bounded dimension;
- acyclic execution and producer/consumer shape compatibility;
- actual file sizes/hashes and ONNX external-data locations;
- cache key sources and invalidation completeness;
- required-device handoff capability and fallback availability; and
- release-state parity and policy completeness.

M0 includes schema self-validation, one valid synthetic fixture and negative
cases for missing checkpoint digest, a bounded profile without a range,
unsafe file paths, incomplete parity evidence and invalid ONNX opset. M1 can
record E1-E3 candidate manifests and fixture hashes with this vocabulary; M2
promotes the reviewed schema and implements manifest-driven runtime dispatch.
