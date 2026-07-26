# M1 E2 — vision output boundary

## Decision

Approve **required-position-only** as the M2 default candidate vision boundary.
Keep **feature-only** as an optional boundary for a profile that explicitly
prefers another 2,654,208 bytes of boundary reduction over the measured median
cost. Keep **full** only as a compatibility fallback; it is not the future
default. This decision does not publish any graph.

## Applicable profiles

The scope and environment are the E1 profile in
[M1_E1_GROUNDING_FUSION.md](M1_E1_GROUNDING_FUSION.md): SAM3 base text-only
image PCS, B1/1008/L32/Q200/FP16, ONNX Runtime CUDA EP and device-resident
IOBinding. The compared consumer is fused `GroundingFull` with raw-200 output.

## Evidence

| Vision boundary | Vision median / p95 | Grounding median / p95 | Combined median / p95 | Boundary bytes | Vision artifact bytes |
|---|---:|---:|---:|---:|---:|
| Full | 278.177 / 279.056 ms | 78.440 / 81.375 ms | 356.617 / 360.431 ms | 111,476,736 | 921,753,595 |
| Required position only | 275.529 / 276.560 ms | 78.354 / 79.625 ms | **353.883 / 356.186 ms** | 58,392,576 | 930,475,898 |
| Feature only | 275.388 / 275.719 ms | 79.024 / 80.364 ms | 354.411 / **356.084 ms** | **55,738,368** | 930,176,938 |

Required-position-only removes 53,084,160 boundary bytes versus full and has
the best median. Feature-only saves a further 2,654,208 bytes but recomputes
the required position tensor in the grounding graph and is 0.528 ms slower at
the combined median. Artifact sizes are recorded observations from this
exporter version, not package-size design constants.

Both reduced boundaries preserved the 0.5 admitted-index set on all fixtures
relative to full. Their outputs were identical to one another; relative to
full, score max-absolute difference was 0.0000101–0.0073630, top-16 box
max-absolute difference 0.000488–0.003662, and top-16 mask IoU
0.991805–0.999340. Capture used the same official eager -> local eager ->
`ExportedProgram` -> ORT chain recorded by E1.

Process-level persistent/peak observations for the three vision sessions were
9,557/9,557, 9,727/9,727 and 9,823/9,823 MiB respectively. Session arenas and
retained device inputs are included, so these values establish comparison
conditions, not portable VRAM minima.

## Replay

Use the E1 command and fixture. The E2 profiles and metrics are emitted in
`.m1-work/measurement_report.json` by harness commit `eb1d42c` from candidate
graphs introduced at `22d0f27`.
