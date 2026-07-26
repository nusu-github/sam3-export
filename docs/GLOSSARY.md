# Export architecture glossary and scope rules

This document fixes the M0 vocabulary used by export code, manifests, runtime
code and release documentation. Longer rationale and hypotheses remain in the
repository-level `SAM3_EXPORT_PARTITIONING_NOTES.md`.

## The four layers

| Layer | Owns | Must not expose or imply |
|---|---|---|
| Public Session / Predictor API | Meaningful operations over images, text, prompts, object IDs and session handles | Backend tensor names, `OrtValue`, memory slots or Multiplex bucket slots |
| Host Runtime / State Store | Decode/preprocess, tokenization, cache keys and invalidation, selection/NMS, dispatch, frame/object/bucket lifecycle, device buffer handoff | Learned computation reimplemented in Python or routine CPU/NumPy round-trips for large graph intermediates |
| Canonical Logical Tensor Components | Tensor-only units for weight mapping, local eager/export parity and unsupported-op isolation | A promise that every component is separately packaged or publicly supported |
| Deployment Plans / Backend Artifacts | Trace-specific composition of components, capture constraints, backend lowering and distributed files | A universal graph for all capabilities or an assumption that module boundaries are deployment cuts |

Components and artifacts are many-to-many. A plan may fuse adjacent
components, and the same semantic plan may have different fused/split recipes
for backends with different device-handoff or operator capabilities.

## Boundary rule

A deployment cut needs at least one concrete reason:

- **Lifetime:** an output is reused after its producer would otherwise rerun;
- **Fan-out:** multiple consumers share the output;
- **Policy:** the host inspects a compact result and chooses later work; or
- **Compatibility:** backend, precision, operator or dynamic-shape constraints
  need isolation.

Class or wrapper boundaries alone are not a deployment-cut reason. Boundary
cost includes materialized bytes, liveness/VRAM, launch/synchronization,
reduced graph fusion, duplicate weights and ABI maintenance.

## Scope labels

Every user-facing artifact, README and plan must identify all of these axes:

| Axis | Allowed vocabulary at M0 |
|---|---|
| Model family | `SAM3 base`, `SAM3.1` |
| Capability | `image PCS`, `interactive image PVS`, `base video tracking`, `multiplex video tracking` |
| Prompt coverage | `text-only`, `geometry/exemplar`, `point/box/mask`, or an explicit combination |
| Lifecycle | `shipped`, `candidate`, `planned`, `test-only`, `deprecated`; legacy is expressed by classification, not folded into lifecycle |
| Dispatch role | `default`, `optional`, `fallback`, `legacy`, `not-applicable`; independent from lifecycle |
| Backend/profile | Backend, execution provider, device handoff, dtype and static/bounded-dynamic shape profile |
| Contract version | Public plan/ABI version, distinct from component class names and manifest schema version |

The exact scope label for the current four-graph bundle is:

> **SAM3 text-only image PCS / legacy split v1**

It must not be described as full SAM3, interactive SAM, video tracking,
SAM3.1 support or the future default recipe.

## Catalog classes

| Class | Meaning | Where documented |
|---|---|---|
| Public deployment artifact | A reviewed, packaged graph or bundle with an explicit semantic contract and release status | `EXPORT_CUTS.md` and a machine-readable manifest |
| Canonical component | A tensor unit used to understand, compose and test a model | `INTERNAL_COMPONENTS.md` |
| Internal fixture / test cut | A wrapper or shape profile built only for local parity/export tests | `INTERNAL_COMPONENTS.md`; never counted as a shipped artifact |
| Legacy shipped artifact | A usable previous package whose precise scope and limitations remain documented | `EXPORT_CUTS.md`, with legacy classification and dispatch role plus shipped lifecycle |
| Deployment plan | A semantic composition and dispatch contract; it may be planned before artifacts exist | `DEPLOYMENT_PLANS.md` |

## Ownership and source-of-truth rules

- `EXPORT_CUTS.md` owns the short public artifact tensor-I/O catalog.
- `INTERNAL_COMPONENTS.md` owns component and fixture classification.
- `DEPLOYMENT_PLANS.md` owns composition, representative traces,
  default/optional/fallback intent and backend dispatch.
- The plan manifest owns the runtime-readable artifact, tensor, cache, state,
  handoff, capture and file-integrity contract.
- Decision records own measurements and adopt/optional/reject decisions.
- README files summarize and link; they do not duplicate the complete tensor
  schema.

Changing a public contract requires its review gate. Updating a hypothesis or
adding an internal fixture does not silently change a shipped plan.
