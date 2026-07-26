# M1 E3 — selected-K mask continuation

## Decision

Approve fixed **K=32** as an **optional M2 policy candidate**. The M2 default
remains fused raw-200 `GroundingFull`; it is also the fallback when CUDA
device-resident continuation is unavailable or a caller cannot accept the
32-result capacity contract. K=16 and K=64 are measured but not admitted to the
initial plan.

Host selection reads compact scores/boxes/presence only. Query embeddings and
other continuation tensors remain CUDA-resident; the host uploads fixed-shape
`selected_indices[1,32]` plus `valid_mask[1,32]`. A CPU/NumPy round trip of the
large continuation is not an allowed fallback.

## Applicable profiles

The scope and environment are the E1 profile in
[M1_E1_GROUNDING_FUSION.md](M1_E1_GROUNDING_FUSION.md). K=32 additionally
means at most 32 post-threshold proposals; unused slots are zeroed by
`valid_mask`. It is not evidence for dynamic K or other capacities.

## Evidence

| Recipe | Median / p95 | Persistent / peak VRAM | Launches with proposals | D2H / H2D bytes | Mask D2H median / p95 |
|---|---:|---:|---:|---:|---:|
| All 200 split baseline | 79.172 / 80.620 ms | 4,913 / 4,913 MiB | 2 | 33,179,602 / 0 | included |
| K=16 | 77.694 / 79.122 ms | 4,933 / 4,933 MiB | 3 | 2,656,210 / 144 | 0.310 / 0.443 ms |
| **K=32** | **76.937 / 77.930 ms** | 4,939 / 4,939 MiB | 3 | 5,310,418 / 288 | 0.565 / 0.795 ms |
| K=64 | 77.634 / 78.893 ms | 5,451 / 5,451 MiB | 3 | 10,618,834 / 576 | 1.378 / 1.446 ms |

K=32 reduces D2H by 27,869,184 bytes versus all-200 and was fastest in this
run despite one additional launch. K=16 has the smallest copy but the fixture
does not justify making a 16-result capacity the initial optional contract.
K=64 copies and retains more without a measured latency advantage over K=32.
These are profile decisions, not universal K constants.

The warmed ORT operator profile for K=32 recorded 0.131 ms for device gather,
2.934 ms for the pixel decoder, 0.391 ms for mask query projection/einsum, and
0.565 ms median for final mask D2H. Node timings are medians after three warmup
runs. Two of five cases had zero admitted proposals and skipped the mask stage,
using two launches rather than three.

For K=16/32/64, all five cases exactly matched the all-200 query scores, top-16
boxes and 0.5 admitted-index sets. Every valid selected mask matched exactly
(max-absolute difference 0, IoU 1); zero-proposal outputs were all zero. Policy
fixtures additionally cover threshold equality, stable tie order and NMS
ordering. The official/eager/export/backend chain is the one recorded by E1.

## Replay

Use the E1 command and fixture. The E3 metrics and per-case selection records
are emitted in `.m1-work/measurement_report.json` by harness commit `eb1d42c`
from candidate graphs introduced at `22d0f27`.
