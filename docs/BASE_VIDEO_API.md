# SAM3 base video Public API

This document is the Public API and `BaseVideoStateV1` contract for the shipped
`sam3_base_video_tracking_ortcuda_v1` plan. Its scope is **SAM3 base video
tracking / point-box-mask correction / per-object batch / ORT CUDA v1**. It
does not change the M2 image PCS or M3 interactive-image defaults and does not
cover streaming input, SAM3.1 bucket state or Multiplex.

## Session API

```python
from sam3.runtime import create_video_session

session = create_video_session(
    "sam3_base_video_tracking_ortcuda_v1",
    bundle_dir="artifacts/sam3-base-video-tracking-ortcuda-v2",
)
video = session.set_video(frames)
session.add_object(42)
preview = session.preview(42, 0, prompt)
# Choose a displayed multimask logit, then request a single-mask correction.
single = session.preview(42, 0, corrected_prompt, single_options)
prediction = session.commit(single.preview_handle)
frames = session.propagate(start_frame=1, end_frame=20)
session.close()
```

`set_video` accepts a non-empty in-memory sequence of PIL images or `uint8`
NumPy arrays. Frames are converted to RGB and must have identical dimensions.
The returned `VideoHandle` is session-bound and opaque. Replacing the video
clears objects, previews and frame caches. Streaming input is outside M4.

`preview` reuses M3's public coordinate and P16/box1/mask288 validity semantics.
Its `VideoPreview` contains display masks, scores, 288x288 low-resolution
logits, metadata and a `PreviewHandle | None`. The default multimask3 result is
display-only and has no handle. Only a single1 result can be committed.
Preview never mutates tracker state; a correction can therefore make repeated
multimask or single-mask previews before committing the final single result.

`commit` consumes a handle once and returns a `VideoPrediction`. A correction
at an already conditioned frame replaces that frame's slot without consuming
capacity, increments the state revision, and invalidates later non-conditioning
entries in the affected direction. All older preview handles then become
stale.

`propagate` processes all active objects in fixed-capacity chunks. Its frame
range is inclusive. Forward uses ascending indices; `reverse=True` requires a
descending range and uses signed relative ages. Each frame returns one
`VideoFramePrediction` containing `object_ids[N]`, `masks[N,H,W]`,
`scores[N]` and public metadata. A frame is encoded once regardless of the
number of objects; B4 handles up to four objects in one tracker launch and
capacity+1 in two launches.

The public dataclasses and handles expose no backend tensor name, device
object, memory slot or `OrtValue`. Public outputs are NumPy values:

- `VideoPreview`: `masks`, `scores`, `low_res_logits`, `preview_handle`,
  `metadata`;
- `VideoPrediction`: `object_id`, `frame_index`, `mask`, `score`,
  `low_res_logits`, `metadata`;
- `VideoFramePrediction`: `frame_index`, `object_ids`, `masks`, `scores`,
  `metadata`.

## `BaseVideoStateV1`

This ABI is fixed/padded per object and is distinct from the planned SAM3.1
bucket-space state.

| Parameter | Shipped value |
|---|---:|
| Mask memories | 7 |
| Conditioning spatial capacity | 4 |
| Non-conditioning spatial capacity | 6 |
| Total spatial input capacity | 10 |
| Object pointer capacity | 16 |
| Hidden / memory dimension | 256 / 64 |
| Memory spatial size | 72x72 |
| Temporal stride | 1 |
| Memory sigmoid scale / bias | 20.0 / -10.0 |
| Non-overlap memory policy | false |

The graph boundary carries object validity, memory feature/position, memory
validity, signed age and conditioning flags, plus object pointers and their
validity/age/conditioning flags. Invalid objects, memories and pointers are
masked from attention. The official memory-0 route is used when no valid
memory exists; one or more memories use memory attention.

Host state owns object ID, frame index, direction, revision, slot selection,
validity, ages and the last low-resolution logits. Large frame, memory and
pointer tensors remain private CUDA values. One object may own at most four
different conditioning frames. Attempting a fifth raises
`StateCapacityError`; there is no silent eviction.

## Cache and device boundary

The frame-cache key includes preprocessed frame bytes, video/frame identity,
original size, checkpoint digest, selected profile,
`memory-aware-frame-view-v1` and the manifest key version. It cannot alias the
M3 initial/no-memory image cache.

Frame features, state, pointers and preview-to-commit values remain CUDA
resident through ORT CUDA IOBinding. Fixed prompt/state arrays are uploaded at
the graph boundary. D2H is limited to public scores, final low-resolution
logits and final masks. CUDA EP and IOBinding are mandatory; there is no CPU or
legacy fallback.

## Errors and lifecycle

The runtime rejects unknown or wrong-scope plans, a missing CUDA/IOBinding
capability, use before `set_video`, invalid videos/frames/prompts/ranges,
duplicate or unknown objects, propagation without conditioning, multimask or
foreign/stale/already-used preview handles, conditioning-capacity overflow,
double close and use after close. These are reported with the public manifest,
capability, session, video, object, preview-handle or capacity exception types
exported from `sam3.runtime`.
