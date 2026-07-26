# Interactive image PVS Public API v1

Owner: **API Owner + Runtime Lead**. This document is the minimal ABI source of
truth for the M3 SAM3 base interactive image PVS session. Plan composition and
dispatch remain owned by [DEPLOYMENT_PLANS.md](DEPLOYMENT_PLANS.md).

```python
import numpy as np

from sam3.runtime import (
    InteractivePredictOptions,
    InteractivePrompt,
    create_interactive_session,
)

session = create_interactive_session(
    "sam3_base_interactive_image_pvs_ortcuda_v1",
    bundle_dir="artifacts/sam3-interactive-image-pvs-ortcuda-v2",
)
image_handle = session.set_image(image)
first = session.predict(
    InteractivePrompt(
        points_xy=np.asarray([[900.0, 580.0]], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int64),
    )
)
selected = first.low_res_logits[int(np.argmax(first.scores))]
second = session.predict(
    InteractivePrompt(
        points_xy=np.asarray([[900.0, 580.0], [1180.0, 710.0]]),
        point_labels=np.asarray([1, 1]),
        mask_logits=selected,
    ),
    InteractivePredictOptions(multimask_output=False),
)
session.close()
```

`set_image` accepts a PIL image or an `uint8` NumPy HWC image and returns the
same public `ImageHandle` type as the image PCS API. The handle exposes only
the cache key and original `(height, width)`. Image encoding uses RGB,
bilinear 1008x1008 resize and `(x / 255 - 0.5) / 0.5` FP16 NCHW input.

## Prompt contract

Public points are float32 `(x, y)` pixel coordinates relative to the original
image. Labels are `0` background or `1` foreground. At most 16 points and one
float32 XYXY box are accepted; capacity overflow and invalid box ordering are
errors rather than truncation. A prior mask must be low-resolution logits with
shape `[288,288]` or `[1,288,288]`, normally one selected
`low_res_logits` result from the previous click. Arbitrary binary masks or
other sizes are not implicitly converted to logits.

The Host Runtime maps public coordinates to the 1008 model frame and packs the
fixed graph ABI:

| Input | Static tensor contract |
|---|---|
| Points | `point_coords[1,16,2]` float32, `point_labels[1,16]` int64, `point_valid[1,16]` bool; invalid labels are `-1` |
| Box | `box_xyxy[1,4]` float32 and `has_box[1]` bool |
| Mask | `mask_input[1,1,288,288]` float32 and `has_mask[1]` bool |

Box corners become labels 2/3 ahead of point tokens. A point-or-box prompt
also carries the official not-a-point sentinel. All remaining sparse tokens
are excluded from the relevant attention key sets by validity. No prompt is
represented by all validity false, label `-1`, zero mask input and false
presence flags rather than an empty tensor axis.

## Options and output

`InteractivePredictOptions.multimask_output` defaults to `True` and dispatches
to the static three-mask artifact. `False` dispatches to the separate static
single-mask artifact, including the official dynamic stability policy.
`mask_threshold` defaults to `0.0`. `output_size` is optional
`(height, width)` and otherwise uses the original image size.

`InteractivePrediction` contains bool `masks[M,H,W]`, float32 `scores[M]`, and
clamped float32 `low_res_logits[M,288,288]`. The Host Runtime bilinearly resizes
logits and applies strict `>` thresholding. Metadata records the public plan,
contract/profile, image cache key, prompt counts, chosen multimask artifact,
sizes and output policy; backend tensor names, `OrtValue` and session slots are
adapter-private.

## Lifecycle and scope

The image feature cache includes preprocessed bytes, original size, checkpoint
digest, static profile, `initial-no-memory` condition ID and key version. A
same-key image does not rerun encoding; successful replacement invalidates the
old device cache. Predict calls always rerun learned prompt/decode work.

Prediction before `set_image`, malformed prompts, unknown/tampered plans,
missing CUDA EP/IOBinding, scope mismatch, double close and use-after-close are
explicit errors. There is no CPU pyramid, legacy split or image PCS fallback.
This session has no video/object/memory state and never performs preview,
memory encode or commit; final memory commit belongs to M4.
