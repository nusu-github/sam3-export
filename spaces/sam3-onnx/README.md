---
title: SAM 3 ONNX
emoji: 🧩
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
short_description: Split-ONNX SAM 3 image segmentation
python_version: "3.12"
startup_duration_timeout: 1h
---

# SAM 3 split-ONNX demo

This ZeroGPU Space runs the four ONNX Runtime CUDA graphs from
[`RamRom/sam3-onnx`](https://huggingface.co/RamRom/sam3-onnx): vision encoding,
text encoding, grounding encoding, and grounding decoding. Upload an image and
enter an open-vocabulary concept such as `dog`, `person`, or `red car`.

The application uses ONNX Runtime IOBinding: intermediate vision, text, and
grounding tensors stay as CUDA `OrtValue`s between the four graphs. Only the
initial NumPy inputs are uploaded and the final decoder outputs are copied back
for mask rendering.

The original SAM 3 model and these derived ONNX files are made available under
the [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE).
