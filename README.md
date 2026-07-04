# ComfyUI-CoreAI

> Apple Core AI models (Neural Engine) as ComfyUI vision nodes — depth,
> SAM, detection, VLM, CLIP, image generation. Zero-friction, ANE-accelerated.

**Status:** Alpha. Architecture + code complete, not yet tested on-device.

## What is this?

A ComfyUI custom node pack that runs Apple Core AI models (`.aimodel` packages)
inside ComfyUI workflows. The bridge is a pre-compiled Swift binary
(`coreai-runner`) embedded in the Python package, spawned as a subprocess,
communicating over HTTP on a Unix domain socket.

**One command install:**
```bash
comfy node install ComfyUI-CoreAI
```

## Why Core AI instead of PyTorch/MPS?

- **Neural Engine (ANE)** runs vision models independently from the GPU
- **True hardware parallelism**: depth/detection on ANE while diffusion runs on GPU
- **Lower memory footprint** (55MB vs ~300MB for Depth Anything 3)
- **Faster** for small vision models (8.6ms RF-DETR Nano on ANE)

## Node Reference (9 nodes)

### Vision — Inference

| Node | Model Family | Input | Output | Latency |
|------|-------------|-------|--------|---------|
| **CoreAI Depth Estimation** | Depth Anything 3 | IMAGE → depth map | IMAGE | ~15ms |
| **CoreAI Object Detection** | RF-DETR, YOLOX | IMAGE → bboxes + annotated | IMAGE + JSON + INT | ~8.6ms |
| **CoreAI Segmentation (SAM 3)** | SAM 3, EfficientSAM3 | IMAGE + text → masks | IMAGE + MASK + JSON | ~550ms |
| **CoreAI Instance Segmentation** | RF-DETR-Seg | IMAGE → instances + masks | IMAGE + JSON | ~10ms |
| **CoreAI Vision-Language (VLM)** | Qwen3-VL, MiniCPM-V | IMAGE + text → caption | STRING | ~191 tok/s |

### Analysis

| Node | Model Family | Input | Output |
|------|-------------|-------|--------|
| **CoreAI CLIP Similarity** | CLIP ViT-B/32 | IMAGE + captions → scores | STRING |

### Generation

| Node | Model Family | Input | Output | Latency |
|------|-------------|-------|--------|---------|
| **CoreAI Image Generation** | FLUX.2 klein 4B | TEXT → image | IMAGE | ~17.4s |

### Utils

| Node | Function |
|------|----------|
| **CoreAI Model Loader** | Browse catalog, pre-download models |
| **CoreAI Health Check** | Device info, thermal state, loaded models |

## Requirements

- Apple Silicon Mac (M1+, M4 recommended)
- macOS 27.0+ (Foundation Models framework)
- ComfyUI running locally

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full system design, wire protocol
- [docs/OPEN_QUESTIONS_RESOLVED.md](docs/OPEN_QUESTIONS_RESOLVED.md) — Technical answers with citations

## License

Apache-2.0
