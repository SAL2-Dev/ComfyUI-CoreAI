# ComfyUI-CoreAI

> Apple Core AI models (Neural Engine) as ComfyUI vision nodes — depth,
> detection, SAM, VLM, CLIP, image generation, and on-device LLM.
> Zero-friction, ANE-accelerated, fully integrated with the Core AI model catalog.

**Status:** Alpha. On-device inference validated on Apple M4 Max / macOS 27.0.

## What is this?

A ComfyUI custom node pack that runs Apple Core AI models (`.aimodel` packages)
inside ComfyUI workflows. The bridge spawns a pre-compiled Swift binary
(`coreai-runner`) as a subprocess on first inference — the user never manages
it manually. Models download from the [Core AI catalog](https://github.com/kevinqz/coreai-catalog)
with a visible progress bar and a one-click Download button on each node.

## Installation

### Via ComfyUI Manager (recommended)

1. Open ComfyUI &rarr; Manager &rarr; Install Custom Nodes
2. Search for "CoreAI"
3. Click Install
4. Restart ComfyUI

The runner binary downloads automatically on first use.

### Manual install

```bash
cd <comfyui>/custom_nodes
git clone https://github.com/kevinqz/ComfyUI-CoreAI.git
cd ComfyUI-CoreAI
pip install -e .
```

Restart ComfyUI. The Core AI runner binary downloads automatically on first
inference — no manual setup required.

## Why Core AI instead of PyTorch/MPS?

- **Neural Engine (ANE)** runs vision models independently from the GPU
- **True hardware parallelism**: depth/detection on ANE while diffusion runs on GPU
- **Lower memory footprint** (55MB vs ~300MB for Depth Anything 3)
- **Faster** for small vision models (8.6ms RF-DETR Nano on ANE)

## Node Reference (10 nodes)

### Vision — Inference

| Node | Model Family | Input | Output | Latency |
|------|-------------|-------|--------|---------|
| **CoreAI Depth Estimation** | Depth Anything 3 | IMAGE &rarr; depth map | IMAGE | ~15ms |
| **CoreAI Object Detection** | RF-DETR, YOLOX | IMAGE &rarr; bboxes + annotated | IMAGE + JSON + INT | ~8.6ms |
| **CoreAI Segmentation (SAM 3)** | SAM 3, EfficientSAM3 | IMAGE + text &rarr; masks | IMAGE + MASK + JSON | ~550ms |
| **CoreAI Instance Segmentation** | RF-DETR-Seg | IMAGE &rarr; instances + masks | IMAGE + JSON | ~10ms |
| **CoreAI Vision-Language (VLM)** | Qwen3-VL, MiniCPM-V | IMAGE + text &rarr; caption | STRING | ~191 tok/s |

### Analysis

| Node | Model Family | Input | Output |
|------|-------------|-------|--------|
| **CoreAI CLIP Similarity** | CLIP ViT-B/32 | IMAGE + captions &rarr; scores | STRING |

### Generation

| Node | Model Family | Input | Output | Latency |
|------|-------------|-------|--------|---------|
| **CoreAI Image Generation** | FLUX.2 klein 4B | TEXT &rarr; image | IMAGE | ~17.4s |

### Apple On-Device (FoundationModels)

| Node | Model Family | Input | Output | Notes |
|------|-------------|-------|--------|-------|
| **CoreAI Apple Text** | Apple Intelligence | TEXT &rarr; text | STRING | No runner needed — uses FoundationModels directly (macOS 26+) |

### Utils

| Node | Function |
|------|----------|
| **CoreAI Model Loader** | Browse catalog, pre-download models |
| **CoreAI Health Check** | Device info, thermal state, loaded models |

## How it works

Every node appears in the ComfyUI canvas with:
- **Model dropdown** — populated live from the Core AI catalog, filtered by capability
- **Status badge** — shows "Not installed · 54.5MB", "Downloading 67%", or "Ready"
- **Download button** — click to install a model without running the workflow

The user drops a node, selects a model, clicks Download if needed, and runs.
Everything else is automatic.

## Requirements

- Apple Silicon Mac (M1+, M4 recommended)
- macOS 27.0+ for vision nodes (Core AI framework)
- macOS 26.0+ for Apple Text node (FoundationModels)
- ComfyUI running locally

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full system design, wire protocol, seamless lifecycle

## License

Apache-2.0
