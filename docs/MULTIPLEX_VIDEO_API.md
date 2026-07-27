# SAM3.1 Multiplex video Public API

This document owns the public state and lifecycle contract for
`sam3_1_multiplex_video_tracking_ortcuda_v1`. The shipped scope is **SAM3.1
multiplex video tracking / point-box-mask correction / bucket16 / ORT CUDA
v1**.

The public surface accepts video frames, public object IDs and point/box/mask
prompts. Bucket/slot assignments, backend tensor names, device values and
`OrtValue` objects are private runtime details.

## Session operations

```python
from sam3.runtime import create_multiplex_video_session
from sam3.runtime.interactive_image import (
    InteractivePredictOptions,
    InteractivePrompt,
)

session = create_multiplex_video_session(
    "sam3_1_multiplex_video_tracking_ortcuda_v1",
    bundle_dir="artifacts/sam3-multiplex-video-tracking-ortcuda-v2",
)
video = session.set_video(frames)
session.add_object(42)
preview = session.preview(
    42,
    0,
    InteractivePrompt(points_xy=[[120.0, 80.0]], point_labels=[1]),
    InteractivePredictOptions(multimask_output=False),
)
prediction = session.commit(preview.preview_handle)
tracked = session.propagate(start_frame=1, end_frame=20)
session.remove_object(42)
session.close()
```

`MultiplexVideoSession` provides `set_video`, `add_object`, `remove_object`,
`preview`, `commit`, `propagate` and `close`. It reuses the existing public
interactive prompt/options and video output dataclasses. Replacing the video
clears objects, previews, frame caches and private bucket state.

`preview` is non-mutating. Multimask output is display-only; only a single-mask
preview has a commit handle. `commit` consumes that handle exactly once and
scatter-replaces only the selected object's slot before encoding its
conditioning memory. A commit increments the private mutation revision but
does not change object-to-slot assignment revision.

`propagate` requires every active object to have a committed conditioning
frame. Its inclusive range is ascending for forward propagation and descending
when `reverse=True`. Results are demuxed only at the final public boundary and
are ordered by ascending public object ID.

## `MultiplexStateV1`

The host owns only public identity and lifecycle:

- an integer object ID and its private bucket/slot assignment;
- add, remove and in-place replacement semantics;
- assignment revision and deterministic public result order.

New objects take the lowest free slot in the first bucket, then the second
bucket. Remove invalidates the slot and makes it reusable. Add/remove/
replacement increments the assignment revision and makes older preview handles
stale. Propagation and correction commit never compact or relocate other
objects and do not increment assignment revision.

The backend owns all large CUDA state: slot validity, shared memory and memory
position, object pointers, current low/high-resolution bucket masks and
scores. `MultiplexStateV1` is incompatible with M4 `BaseVideoStateV1`; neither
state ABI is implicitly converted to the other.

| Parameter | Shipped value |
|---|---:|
| Slot capacity per bucket | 16 |
| Maximum buckets / objects | 2 / 32 |
| Conditioning / non-conditioning spatial capacity | 4 / 6 |
| Total spatial input capacity | 10 |
| Pointer frame capacity | 16 |
| Hidden / memory dimension | 256 / 256 |
| Memory spatial size | 72x72 |
| Image / low-resolution mask size | 1008 / 288 |
| Memory sigmoid scale / bias | 2.0 / -1.0 |

These are checkpoint/official-builder values verified by the M5 adapter, not
defaults inherited from the base-video implementation.

## Dispatch and device boundary

The fixed B1 artifact processes one native 16-slot bucket. The host dispatches
it once for 1–16 objects and twice for 17–32 objects. The two calls keep
independent bucket trajectories; normal propagation passes each bucket's CUDA
state directly to the next frame.

Tri interactive and propagation frame views are encoded once per frame and
cached as CUDA values. Shared memory, pointer and frame handoff use ORT CUDA
IOBinding. There is no CUDA EP fallback node and no CPU fallback. Large state
has no routine D2H/H2D, demux or remux; only public masks, scores,
low-resolution logits and metadata cross D2H at the output boundary.

## Handles, outputs and errors

`VideoPreview`, `VideoPrediction` and `VideoFramePrediction` preserve the
existing public NumPy semantics. Metadata may name the public plan, contract,
profile, state ABI, object/frame identity and assignment revision, but never a
slot, backend tensor or device object.

The runtime deterministically rejects:

- unknown/wrong-scope plans or missing CUDA/IOBinding capability;
- use before `set_video`, invalid video/frame/prompt/range or propagation
  direction;
- duplicate/unknown object IDs, more than 32 objects, or propagation before
  every active object is conditioned;
- foreign, stale, unknown, multimask or already-used preview handles;
- double close and use after close.

The manifest owns exact tensor bindings, checkpoint-derived parameters,
cache/state compatibility and file hashes. Profile adoption and rejected
candidates are recorded in
[M5_SAM31_MULTIPLEX_PROFILE.md](decision-records/M5_SAM31_MULTIPLEX_PROFILE.md).
