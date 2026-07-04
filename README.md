# ComfyUI-CoreAI

> Core AI models (Neural Engine) as ComfyUI vision nodes — depth, SAM,
> detection, VLM, image generation. Zero-friction, ANE-accelerated.

**Status:** Architecture phase. Not yet functional.

## What is this?

A ComfyUI custom node that runs Apple Core AI models (`.aimodel` packages)
inside ComfyUI workflows. The bridge is a pre-compiled Swift binary
(`coreai-runner`) embedded in the Python package, spawned as a subprocess,
communicating over HTTP on a Unix domain socket.

**One command install:**
```bash
comfy node install ComfyUI-CoreAI
```

**Why Core AI instead of PyTorch/MPS?**
- Neural Engine (ANE) runs vision models independently from the GPU
- True hardware parallelism: depth/detection on ANE while diffusion runs on GPU
- Lower memory footprint (55MB vs ~300MB for Depth Anything 3)
- Faster for small vision models (8.6ms RF-DETR Nano on ANE)

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full system design, wire protocol, node catalog
- [docs/OPEN_QUESTIONS_RESOLVED.md](docs/OPEN_QUESTIONS_RESOLVED.md) — Technical answers with primary source citations

## Requirements

- Apple Silicon Mac (M1+, M4 recommended)
- macOS 27.0+ (Foundation Models framework)
- ComfyUI running locally

## License

Apache-2.0 (matching CoreAIKit and the upstream model licenses)
