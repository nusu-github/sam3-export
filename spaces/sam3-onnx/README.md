---
title: SAM3 ONNX Runtime M2–M4
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
short_description: SAM3 image and video ONNX Runtime demos
python_version: "3.12"
startup_duration_timeout: 1h
---

# SAM3 ONNX Runtime M2–M4

This ZeroGPU Space runs the released [`RamRom/sam3-onnx`](https://huggingface.co/RamRom/sam3-onnx)
ONNX Runtime CUDA bundles through manifest-validated Host Runtime sessions.

- **M2** — SAM3 base text-only image PCS.
- **M3** — SAM3 base interactive image PVS with point, box, and prior mask-logit prompts.
- **M4** — SAM3 base video tracking with per-object state, preview, commit, and batched propagation.

All graphs use CUDA IOBinding: frame/image features, prompt intermediates, and video memory
remain device-resident; only public results return to the host. The video tab samples uploads to
at most 60 frames and supports five objects. It is an in-memory video demo, not a streaming service.

The exact Hub revision is `release-m2-m4-ortcuda-v1`. It requires ONNX Runtime CUDA and has no
CPU or legacy fallback. SAM3.1 Tri neck, bucket-space state, and Multiplex are not included.

The original SAM 3 model and these derived ONNX files are available under the
[SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE).
