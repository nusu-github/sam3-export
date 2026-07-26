---
title: SAM3 Legacy Split ONNX
emoji: 🧩
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
short_description: SAM3 text-only image PCS, legacy split v1
python_version: "3.12"
startup_duration_timeout: 1h
---

# SAM3 text-only image PCS / legacy split v1 demo

This ZeroGPU Space runs the four ONNX Runtime CUDA graphs from
[`RamRom/sam3-onnx`](https://huggingface.co/RamRom/sam3-onnx): vision encoding,
text encoding, grounding encoding, and grounding decoding. Upload an image and
enter an open-vocabulary concept such as `dog`, `person`, or `red car`.

The application uses ONNX Runtime IOBinding: intermediate vision, text, and
grounding tensors stay as CUDA `OrtValue`s between the four graphs. Only the
initial NumPy inputs are uploaded and the final decoder outputs are copied back
for mask rendering.

The exact scope is **SAM3 text-only image PCS / legacy split v1**. This demo
uses ONNX Runtime CUDA EP + IOBinding with fp16 tensors, batch 1, a 1008x1008
model image and text length 32. It does not cover geometry/exemplar prompts,
semantic output, production interactive PVS, video tracking, or SAM3.1. It
thresholds the 200 detector queries, sorts matches by score, and renders at
most 25 without NMS. The split is a supported legacy recipe and comparison
baseline, not the future default deployment plan.

The original SAM 3 model and these derived ONNX files are made available under
the [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE).
