# Image PCS Public API v1

Owner: **API Owner + Runtime Lead**. This document is the minimal ABI source of
truth for the M2 SAM3 base text-only image PCS session. Scope vocabulary and
exclusions remain owned by [GLOSSARY.md](GLOSSARY.md), while plan composition
and dispatch remain owned by [DEPLOYMENT_PLANS.md](DEPLOYMENT_PLANS.md).

```python
from sam3.runtime import PredictOptions, create_image_session

session = create_image_session(
    "sam3_base_image_pcs_text_ortcuda_v1",
    bundle_dir="artifacts/sam3-image-pcs-ortcuda-v2",
)
image_handle = session.set_image(image)
prompt_handle = session.set_text("a truck")
prediction = session.predict_text(PredictOptions(score_threshold=0.5))
session.close()
```

`set_image` accepts an RGB-convertible PIL image or an `uint8` NumPy HWC image.
`set_text` accepts one UTF-8 Python string. Public handles contain only cache
identity and meaningful size/token metadata; backend tensor names, `OrtValue`
and session slots are adapter-private.

## Prediction options and output

| Field | Contract |
|---|---|
| `score_threshold` | Default `0.5`; admission uses strict `>` after `sigmoid(logit) * sigmoid(presence)` |
| `nms_iou_threshold` | Optional mask-IoU NMS threshold in `[0,1]` |
| `max_results` | Optional non-negative result limit; the K=32 plan rejects values above 32 |
| `output_size` | Optional `(height, width)`; otherwise the original image size |

`Prediction.boxes_xyxy` is float32 pixel-space XYXY. `scores` is float32.
`masks[N,H,W]` is float32 probability after sigmoid and bilinear resize.
`metadata` records the resolved plan/contract/profile, image and prompt cache
keys, output policy, and original/output sizes.

## Lifecycle and errors

Image and prompt caches are independent. Repeating the same normalized image
or token sequence does not rerun its encoder; changing one replaces only that
cache. Prediction requires both caches. `close()` releases per-session caches
and backend sessions. Missing state, unknown/tampered manifests, missing CUDA
IOBinding capability, double close, and use-after-close raise explicit runtime
exceptions; there is no implicit dispatch to the legacy v1 bundle or a CPU
pyramid handoff.

The shipped plan IDs are the fused raw-200 default
`sam3_base_image_pcs_text_ortcuda_v1`, optional fixed-K policy
`sam3_base_image_pcs_text_ortcuda_selected_k32_v1`, and corrected split
fallback `sam3_base_image_pcs_text_ortcuda_split_v1`. This ABI does not add
interactive, video, geometry/exemplar, semantic, or SAM3.1 operations.
