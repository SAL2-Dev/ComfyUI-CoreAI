# ComfyUI-CoreAI — Architecture Document

> **State-of-the-art custom node for running Apple Core AI models inside ComfyUI.**
> Zero-friction install, Neural Engine acceleration, vision-first.

---

## Executive Summary

ComfyUI-CoreAI bridges two runtimes that have never been connected:
**Python/PyTorch (ComfyUI)** and **Swift/Foundation Models (Core AI)**.

The bridge is a **pre-compiled Swift binary** (`coreai-runner`) embedded in
the Python package, spawned as a subprocess, communicating over **HTTP on a
Unix domain socket**. The user never sees, configures, or installs the
runner separately.

```
comfy node install ComfyUI-CoreAI  →  done
```

---

## Design Principles

| Principle | Rule |
|-----------|------|
| **Zero friction** | One command install. Binary auto-downloads. No Swift toolchain needed. No server to manage. |
| **Invisible runtime** | The Swift runner is spawned/killed by ComfyUI lifecycle. User never interacts with it. |
| **Vision-first** | Depth, SAM, detection, VLM — the nodes that matter for generative workflows. |
| **Catalog-native** | Model dropdowns populated from the coreai-catalog API. Not hardcoded lists. |
| **ANE by default** | Core AI's Neural Engine scheduling is the default. No manual GPU config. |
| **Fail loudly** | If a model needs a patch or isn't supported on the device, say so clearly in the node. |

---

## System Architecture

### Process Model

```
┌─────────────────────────────────────────────────────────┐
│  ComfyUI Process (Python)                                │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ DepthNode    │  │ SAMNode      │  │ DetectionNode│   │
│  │ (Python)     │  │ (Python)     │  │ (Python)     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │            │
│         └────────┬────────┴────────┬────────┘            │
│                  │                 │                      │
│         ┌────────▼─────────┐      │                      │
│         │  bridge.py       │      │                      │
│         │  (subprocess     │      │                      │
│         │   lifecycle)     │      │                      │
│         └────────┬─────────┘      │                      │
│                  │                 │                      │
│     ────────────┼─────────────────┼──────────────────    │
│     PROCESS BOUNDARY (socket)     │                      │
│     ────────────┼─────────────────┼──────────────────    │
│                  │                 │                      │
│         ┌────────▼─────────────────▼─────────┐           │
│         │     coreai-runner (Swift)           │           │
│         │     Embeds CoreAIKit SPM            │           │
│         │                                      │           │
│         │  ┌─────────┐  ┌──────┐  ┌────────┐ │           │
│         │  │HTTP SVR │  │Model │  │Catalog │ │           │
│         │  │(Unix    │  │Cache │  │Client  │ │           │
│         │  │ socket) │  │      │  │(lazy)  │ │           │
│         │  └────┬────┘  └──┬───┘  └────────┘ │           │
│         │       │          │                   │           │
│         │  ┌────▼──────────▼───────────────┐  │           │
│         │  │   Foundation Models Framework  │  │           │
│         │  │   ┌─────────────────────────┐  │  │           │
│         │  │   │  CoreAIRunner           │  │  │           │
│         │  │   │  CoreAIImageSegmenter   │  │  │           │
│         │  │   │  CoreAIDiffusionPipeline│  │  │           │
│         │  │   └────────────┬────────────┘  │  │           │
│         │  │                │               │  │           │
│         │  │   ┌────────────▼────────────┐  │  │           │
│         │  │   │  Apple Silicon Hardware │  │  │           │
│         │  │   │  ANE │ GPU │ CPU        │  │  │           │
│         │  │   └─────────────────────────┘  │  │           │
│         │  └────────────────────────────────┘  │           │
│         └──────────────────────────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Why HTTP over Unix Socket?

| Option | Latency | Complexity | Debuggability | Verdict |
|--------|---------|------------|---------------|---------|
| Unix socket + HTTP/JSON | ~0.3ms | Low | curl-compatible | **CHOSEN** |
| stdin/stdout pipes | ~0.1ms | Medium (framing) | Hard | Rejected — no concurrent requests |
| TCP HTTP | ~1ms | Low | Easy | Rejected — port conflicts |
| Shared memory | ~0.05ms | Very high | Hard | Rejected — premature optimization |
| gRPC | ~0.3ms | High | Medium | Rejected — protobuf overhead |

Unix socket + HTTP gives:
- **Zero port conflicts** (socket file, not TCP port)
- **curl-debuggable** (`curl --unix-socket /tmp/coreai.sock /v1/models`)
- **Concurrent requests** (HTTP server handles parallel inference)
- **Clean shutdown** (socket file cleaned on process exit)

---

## Wire Protocol

Base URL: `http://unix` (Unix socket at `/tmp/coreai-runner.sock`)

### Health

```
GET /v1/health

→ 200 {
    "status": "healthy",
    "device": "Apple M4 Pro",
    "chip": "Apple M4 Pro (14-Core CPU, 20-Core GPU, 16-Core Neural Engine)",
    "memory_total_gb": 48,
    "memory_available_gb": 31.2,
    "macos_version": "26.6",
    "coreai_version": "26.6",
    "loaded_models": ["depth-anything-3-small"],
    "thermal_state": "nominal"
  }
```

### List Models

```
GET /v1/models?capability=monocular-depth

→ 200 {
    "models": [
      {
        "id": "depth-anything-3-small",
        "name": "Depth Anything 3 Small",
        "family": "Depth Anything",
        "capability": "monocular-depth",
        "size_mb": 54.5,
        "precision": "fp16",
        "license": "Apache-2.0",
        "commercial_use": "likely",
        "device_support": ["iphone", "mac"],
        "installed": true,
        "loaded": false,
        "benchmark_ms": 15.2,
        "runner": "CoreAIRunner"
      },
      ...
    ]
  }
```

### Predict (single inference)

```
POST /v1/predict

{
  "model_id": "depth-anything-3-small",
  "input": {
    "image_path": "/tmp/comfyui_input_abc123.png"
  },
  "options": {
    "compute_unit": "auto"       // "auto" | "cpuAndGPU" | "cpuAndNeuralEngine" | "all"
  }
}

→ 200 {
    "model_id": "depth-anything-3-small",
    "output": {
      "depth_map_path": "/tmp/coreai_output_xyz789.png",
      "format": "grayscale16",
      "width": 518,
      "height": 518
    },
    "timing": {
      "load_ms": 0,         // 0 if already cached
      "preprocess_ms": 2.1,
      "inference_ms": 14.8,
      "postprocess_ms": 1.3,
      "total_ms": 18.2
    },
    "compute_unit_used": "GPU"
  }
```

### SAM-style (promptable segmentation)

```
POST /v1/predict

{
  "model_id": "official-sam-3",
  "input": {
    "image_path": "/tmp/input.png",
    "prompts": {
      "points": [
        {"x": 320, "y": 240, "label": "foreground"},
        {"x": 100, "y": 80, "label": "background"}
      ],
      "boxes": [
        {"x1": 50, "y1": 50, "x2": 400, "y2": 400}
      ],
      "text": "a red car"      // optional text prompt
    }
  },
  "options": {
    "compute_unit": "auto",
    "mask_threshold": 0.5,
    "max_masks": 5
  }
}

→ 200 {
    "model_id": "official-sam-3",
    "output": {
      "masks": [
        {
          "mask_path": "/tmp/coreai_mask_0.png",
          "score": 0.97,
          "bbox": {"x1": 52, "y1": 48, "x2": 398, "y2": 402}
        }
      ]
    },
    "timing": { ... }
  }
```

### VLM (vision-language)

```
POST /v1/predict

{
  "model_id": "qwen3-vl-2b",
  "input": {
    "image_path": "/tmp/input.png",
    "prompt": "Describe this image in detail for a Stable Diffusion prompt.",
    "max_tokens": 200,
    "temperature": 0.7
  }
}

→ 200 {
    "model_id": "qwen3-vl-2b",
    "output": {
      "text": "a serene mountain landscape at golden hour, snow-capped peaks...",
      "tokens_generated": 48
    },
    "timing": { ... }
  }
```

### Detection

```
POST /v1/predict

{
  "model_id": "rf-detr-nano",
  "input": {
    "image_path": "/tmp/input.png",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.4
  }
}

→ 200 {
    "model_id": "rf-detr-nano",
    "output": {
      "detections": [
        {"class": "person", "class_id": 0, "score": 0.94,
         "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 450}},
        {"class": "car", "class_id": 2, "score": 0.88,
         "bbox": {"x1": 400, "y1": 200, "x2": 700, "y2": 400}}
      ],
      "num_detections": 2
    },
    "timing": { ... }
  }
```

### Model Lifecycle

```
POST /v1/models/{model_id}/load
POST /v1/models/{model_id}/unload
GET  /v1/models/{model_id}/status
```

### Errors

```
→ 4xx {
    "error": {
      "code": "MODEL_NOT_INSTALLED",
      "message": "Model 'depth-anything-3-small' is not installed. Run the download from the node.",
      "model_id": "depth-anything-3-small",
      "download_url": "https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI/resolve/main/small/da3-small_float16.aimodel"
    }
  }

→ 5xx {
    "error": {
      "code": "INFERENCE_FAILED",
      "message": "The model produced an unexpected tensor shape.",
      "detail": "Expected [1,3,518,518], got [1,3,256,256]. Check image preprocessing."
    }
  }
```

**Error codes:** `MODEL_NOT_INSTALLED` | `MODEL_LOAD_FAILED` | `INFERENCE_FAILED` | `UNSUPPORTED_DEVICE` | `PATCH_REQUIRED` | `MEMORY_INSUFFICIENT` | `TIMEOUT` | `INVALID_INPUT`

---

## Node Catalog (ComfyUI Nodes)

### Phase 1 — Core Vision (MVP)

| Node | Capability | Models | Input → Output | ComfyUI Type |
|------|-----------|--------|---------------|-------------|
| **CoreAI Depth** | monocular-depth | Depth Anything 3 (S/B) | IMAGE → IMAGE (depth) | Preprocessor |
| **CoreAI Segment (SAM)** | promptable-segmentation | SAM 3, EfficientSAM3 | IMAGE + points/boxes → MASK | Mask |
| **CoreAI Detect** | object-detection | RF-DETR (N/S/M/L), YOLOX-S | IMAGE → BBOXES + IMAGE (annotated) | Utility |
| **CoreAI VLM** | vision-language | Qwen3-VL 2B, MiniCPM-V 4.6 | IMAGE + TEXT → TEXT | Prompt |
| **CoreAI Image Gen** | image-generation | FLUX.2 klein 4B, Z-Image Turbo | TEXT → IMAGE | Generator |
| **CoreAI Loader** | — | (all models) | — → MODEL_REF | Loader |

### Phase 2 — Extended

| Node | Capability | Models |
|------|-----------|--------|
| **CoreAI Instance Seg** | instance-segmentation | RF-DETR-Seg (6 variants) |
| **CoreAI Embedding** | image-text-similarity | CLIP ViT-B/32 |
| **CoreAI Doc Retrieve** | visual-document-retrieval | ColModernVBERT |
| **CoreAI STT** | speech-to-text | Whisper large-v3-turbo |

### Node UI Design

Each node follows ComfyUI conventions but adds Core AI-specific UI:

```
┌─────────────────────────────────────────────┐
│  CoreAI Depth Estimation                     │
├─────────────────────────────────────────────┤
│  image    ◄───────────  [IMAGE]              │
│                                              │
│  model    [Depth Anything 3 Small      ▼]    │
│           ⓘ 54.5 MB · fp16 · 15ms · ANE     │
│                                              │
│  compute  [Auto (recommended)          ▼]    │
│           ○ Auto  ○ GPU+CPU  ○ ANE+CPU       │
│                                              │
│  ☐ Keep model loaded between runs            │
│                                              │
│  IMAGE (depth)  ──────►                      │
│  IMAGE (colorized) ───►                      │
└─────────────────────────────────────────────┘
```

The info badge (`ⓘ`) shows real data from the catalog:
- Model size, precision, benchmark latency, optimal compute unit

---

## Runner Architecture (Swift Binary)

### Binary Structure

```
coreai-runner  (single static binary, ~25-30 MB)
│
├── Embedded:
│   ├── Hummingbird HTTP server (lightweight Swift HTTP)
│   ├── CoreAIKit SPM (ALL model loading + inference + download)
│   │   ├── CoreAIKitVision: GraphModel, DepthEstimator, ObjectDetector
│   │   ├── CoreAILM: patched pipelined engine (Qwen3-VL support baked in)
│   │   └── CoreAIKitCore: ModelStore, ModelCatalog, HubClient
│   ├── swift-transformers (HF tokenizer, via CoreAIKit dep chain)
│   └── Catalog API client (lazy fetch, 5min cache)
│
├── NOT included:
│   ├── Models themselves (downloaded on demand via ModelStore)
│   ├── Tokenizers (bundled with each .aimodel package)
│   └── Xcode toolchain (pre-compiled binary)
│
└── Platform:
    ├── arm64-apple-macos27+  (primary — CoreAIKit requires macOS 27.0+)
    └── arm64-apple-ios27+    (future, for server mode)
```

### Model Cache Strategy

```
┌─────────────────────────────────────────┐
│  Model Cache (in coreai-runner)         │
│                                         │
│  State Machine per model:               │
│                                         │
│  NOT_INSTALLED                          │
│       │                                 │
│       │ POST /load  → download .aimodel │
│       ▼                                 │
│  DOWNLOADING (progress via /status)     │
│       │                                 │
│       ▼                                 │
│  INSTALLED                              │
│       │                                 │
│       │ POST /load → load into memory   │
│       ▼                                 │
│  LOADING                                │
│       │                                 │
│       ▼                                 │
│  READY ──────── POST /predict ──────►  │
│       │                                 │
│       │ POST /unload or LRU evict       │
│       ▼                                 │
│  INSTALLED                              │
└─────────────────────────────────────────┘
```

**LRU eviction:** When memory is tight (checked via `ProcessInfo.thermalState` and available RAM), the runner unloads the least-recently-used model. The node can mark models as "sticky" (never evict during a workflow).

**Concurrent models:** Typically 1-3 loaded simultaneously (depth: 55MB, RF-DETR nano: 108MB, YOLOX-S: ~10MB are light; SAM 3 at 1.7GB and FLUX.2 at 4GB are heavy and benefit from explicit unload).

### Catalog Integration

The runner fetches model metadata from the coreai-catalog API at runtime:

```
GET https://coreai-catalog.nousresearch.com/v1/models/{id}
→ JSON: capabilities, modalities, runner_type, device_support, benchmark, provenance
```

This means:
- **New models appear automatically** — no node update needed
- **Model dropdowns are always current** — filtered by capability + device
- **Download URLs resolved dynamically** — from provenance.huggingface
- **5-minute local cache** — avoids hitting the API on every node render

---

## Python Bridge (bridge.py)

### Lifecycle Management

```python
class CoreAIRunner:
    """
    Manages the coreai-runner subprocess lifecycle.
    Singleton — one runner instance per ComfyUI session.
    """

    BINARY_NAME = "coreai-runner"
    SOCKET_PATH = "/tmp/coreai-runner.sock"
    STARTUP_TIMEOUT = 5.0
    SHUTDOWN_TIMEOUT = 3.0

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._binary_path: Path | None = None
        self._client: httpx.Client | None = None
        self._atexit_registered = False

    def ensure_running(self) -> None:
        """Lazy-start the runner on first inference call."""
        if self._process and self._is_alive():
            return

        self._binary_path = self._resolve_binary()
        self._process = subprocess.Popen(
            [str(self._binary_path), "--socket", self.SOCKET_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "COREAI_LOG_LEVEL": "info"},
        )

        # Wait for socket to appear
        self._wait_for_socket()
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=self.SOCKET_PATH),
            timeout=300.0,  # long timeout for big models
        )

        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def shutdown(self) -> None:
        """Clean shutdown — sends SIGTERM, waits, then SIGKILL if needed."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=self.SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if os.path.exists(self.SOCKET_PATH):
            os.unlink(self.SOCKET_PATH)

    def predict(self, model_id: str, input_data: dict, **opts) -> dict:
        """Single inference call."""
        self.ensure_running()
        resp = self._client.post("/v1/predict", json={
            "model_id": model_id,
            "input": input_data,
            "options": opts,
        })
        resp.raise_for_status()
        return resp.json()

    def list_models(self, capability: str | None = None) -> list[dict]:
        """Get available models for a capability."""
        self.ensure_running()
        params = {"capability": capability} if capability else {}
        resp = self._client.get("/v1/models", params=params)
        return resp.json()["models"]
```

### Binary Resolution

```python
def _resolve_binary(self) -> Path:
    """
    Find the coreai-runner binary.
    Priority:
      1. COREAI_RUNNER_PATH env var (dev/override)
      2. Package-bundled binary: <package>/bin/coreai-runner
      3. Download on first run (arm64 macOS only)
    """
    # 1. Env override
    if env_path := os.environ.get("COREAI_RUNNER_PATH"):
        return Path(env_path)

    # 2. Bundled
    pkg_dir = Path(__file__).parent / "bin"
    binary = pkg_dir / self.BINARY_NAME

    if binary.exists():
        return binary

    # 3. Auto-download
    arch = platform.machine()  # arm64 on Apple Silicon
    os_name = platform.system()  # Darwin

    if arch != "arm64" or os_name != "Darwin":
        raise RuntimeError(
            f"ComfyUI-CoreAI requires Apple Silicon (arm64 macOS). "
            f"Detected: {arch} {os_name}"
        )

    version = self._get_runner_version()
    url = f"https://github.com/kevinsaltarelli/ComfyUI-CoreAI/releases/download/{version}/coreai-runner-arm64-macos"

    binary.parent.mkdir(parents=True, exist_ok=True)
    download_with_progress(url, binary)
    os.chmod(binary, 0o755)
    return binary
```

---

## Repository Structure

```
ComfyUI-CoreAI/
│
├── README.md                    # Install, usage, node reference
├── ARCHITECTURE.md              # This document
├── LICENSE                      # Apache-2.0
├── pyproject.toml               # Python package metadata
├── requirements.txt             # httpx, pillow, numpy (minimal)
│
├── comfyui_coreai/              # Python package (the node)
│   ├── __init__.py              # ComfyUI node registration
│   ├── bridge.py                # Subprocess lifecycle + HTTP client
│   ├── catalog.py               # Catalog API client (cached)
│   ├── image_utils.py           # ComfyUI tensor ↔ file conversion
│   ├── nodes/
│   │   ├── depth.py             # CoreAI Depth Estimation node
│   │   ├── segmentation.py      # CoreAI Segment (SAM) node
│   │   ├── detection.py         # CoreAI Detect (RF-DETR/YOLOX) node
│   │   ├── vlm.py               # CoreAI VLM node
│   │   ├── image_gen.py         # CoreAI Image Generation node
│   │   └── loader.py            # CoreAI Model Loader / downloader
│   ├── bin/                     # Downloaded Swift binary lives here
│   │   └── .gitkeep
│   └── install.py               # comfy-cli post-install hook
│
├── coreai-runner/               # Swift binary source
│   ├── Package.swift            # SPM manifest (links CoreAIKit)
│   ├── Sources/
│   │   └── CoreAIRunner/
│   │       ├── main.swift       # Entry point, arg parsing
│   │       ├── server.swift     # HTTP server (Hummingbird)
│   │       ├── routes.swift     # /v1/* route handlers
│   │       ├── model_cache.swift # Load/unload/LRU (actor-isolated)
│   │       ├── catalog_client.swift # Fetch model metadata
│   │       ├── adapters/        # Thin wrappers over CoreAIKit types
│   │       │   ├── depth.swift     # → CoreAIKitVision.DepthEstimator
│   │       │   ├── detection.swift # → CoreAIKitVision.ObjectDetector
│   │       │   ├── vlm.swift       # → CoreAIKit.KitVisionModel
│   │       │   ├── diffusion.swift # → CoreAIDiffusionPipeline (system)
│   │       │   └── segmenter.swift # → CoreAIImageSegmenter (system)
│   │       └── utils/
│   │           ├── image.swift     # CGImage ↔ file I/O
│   │           ├── device.swift    # Chip detection, thermal state
│   │           └── logging.swift   # Structured logging
│   └── Tests/
│       └── CoreAIRunnerTests/
│           └── ...
│
├── .github/
│   └── workflows/
│       ├── build-runner.yml     # Build Swift binary → GitHub Release
│       ├── test-python.yml      # pytest the Python nodes
│       └── release.yml          # Tag → publish PyPI + GitHub Release
│
└── docs/
    ├── NODE_REFERENCE.md        # Every node, every input/output
    ├── DEVELOPMENT.md           # How to build the Swift binary
    └── BENCHMARKS.md            # Core AI vs PyTorch on same models
```

---

## Model Coverage Matrix

### Phase 1 Nodes (vision — 16 models from catalog)

| Node | Models | Count | Sizes | Latency |
|------|--------|-------|-------|---------|
| **Depth** | Depth Anything 3 Small, Base | 2 | 55-202MB | 15ms |
| **SAM** | SAM 3, EfficientSAM3-TinyViT | 2 | ~1.7GB / unknown | 550ms |
| **Detect** | RF-DETR (N/S/M/L), YOLOX-S | 5 | 10-108MB | 8.6ms |
| **VLM** | Qwen3-VL 2B, MiniCPM-V 4.6 | 2 | 2.3GB | 33-191 tok/s |
| **Image Gen** | FLUX.2 klein 4B, Z-Image Turbo | 2 | 4GB / unknown | 17.4s |

### Phase 2 Nodes (extended — 12 models)

| Node | Models | Count |
|------|--------|-------|
| **Instance Seg** | RF-DETR-Seg (6 variants) | 6 |
| **Embedding** | CLIP ViT-B/32 | 1 |
| **Doc Retrieve** | ColModernVBERT | 1 |
| **STT** | Whisper large-v3-turbo | 1 |

### NOT included (text LLMs — out of scope)

| Capability | Why excluded |
|-----------|-------------|
| chat / text-generation | Ollama/Exo/LM Studio dominate this. No value-add from Core AI path in ComfyUI. |
| embedding (text-only) | Overlaps with ComfyUI's existing text encoders. |
| reranking | Not a ComfyUI use case. |

---

## Adapter Pattern (thin wrappers over CoreAIKit)

CoreAIKit already provides typed pipelines for every model family. Our runner
does NOT re-implement model loading — it wraps CoreAIKit types behind a uniform
async API, adapting inputs/outputs to the wire protocol.

```swift
// Each adapter wraps a CoreAIKit type, translating HTTP JSON ↔ Swift calls

protocol ModelAdapter {
    func predict(_ input: AdapterInput) async throws -> AdapterOutput
}

// Depth: CoreAIKitVision.DepthEstimator
struct DepthAdapter: ModelAdapter {
    let estimator: DepthEstimator  // from CoreAIKitVision
    func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let cgImage = try loadCGImage(from: input.imagePath)
        let depth = try await estimator.estimateDepth(for: cgImage)
        return .depthMap(depth.cgImage())
    }
}

// Detection: CoreAIKitVision.ObjectDetector
struct DetectionAdapter: ModelAdapter {
    let detector: ObjectDetector   // from CoreAIKitVision
    func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let cgImage = try loadCGImage(from: input.imagePath)
        let dets = try await detector.detect(in: cgImage, scoreThreshold: 0.5)
        return .detections(dets)
    }
}

// VLM: CoreAIKit.KitVisionModel (via LanguageModelSession)
struct VLMAdapter: ModelAdapter {
    let model: KitVisionModel      // from CoreAIKit
    let session: LanguageModelSession
    func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let cgImage = try loadCGImage(from: input.imagePath)
        let reply = try await session.respond(to: Prompt {
            input.prompt
            Attachment(cgImage)
        })
        return .text(reply.content)
    }
}

// Diffusion: CoreAIDiffusionPipeline (system framework, via PipelineDescriptor)
struct DiffusionAdapter: ModelAdapter {
    let pipeline: any DiffusionPipeline
    func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let config = PipelineConfiguration(prompt: input.prompt, ...)
        let result = try await pipeline.generateImages(configuration: config) { _ in true }
        return .image(result.images.first!)
    }
}

// Segmentation: CoreAIImageSegmenter (system framework)
struct SegmenterAdapter: ModelAdapter {
    let segmenter: CoreAIImageSegmenter  // from system CoreAI
    func predict(_ input: AdapterInput) async throws -> AdapterOutput {
        let cgImage = try loadCGImage(from: input.imagePath)
        let masks = try await segmenter.segment(cgImage, prompt: input.textPrompt)
        return .masks(masks)
    }
}
```

**Why adapters, not re-implementation:**
- CoreAIKit handles: download, cache, tokenizer, preprocessing, patched engine
- Adapters handle: file I/O, JSON ↔ Swift, error formatting, timing
- Zero duplication of model-specific logic

### Runtime Flags Handling

The catalog specifies per-model runtime flags that the runner must respect:

| Flag | Meaning | Runner Action |
|------|---------|---------------|
| `processor_required` | Needs image processor | Load processor from .aimodel package before inference |
| `tokenizer_required` | Needs tokenizer | Load tokenizer.json from package |
| `patch_required` | Non-stock runtime, needs patches | Apply patches from package before loading |
| `aot_required` | Ahead-of-time compilation | Trigger AOT compile on first load (warn user about delay) |
| `stock_runtime` | Uses standard Apple runner | No modifications needed |

---

## Image Transfer Strategy

ComfyUI tensors (CHW float32) must become Core AI inputs (CVPixelBuffer / CGImage).

**Chosen: file-based transfer** (simplest, debuggable, no shared memory complexity)

```
ComfyUI tensor [1,3,H,W] float32
    │
    ▼  bridge.py
PNG file /tmp/coreai_in_{uuid}.png
    │
    ▼  Unix socket HTTP (just the path string in JSON)
    │
coreai-runner receives path
    │
    ▼  Swift: CGImage → CVPixelBuffer
Foundation Models inference
    │
    ▼  Swift: result → PNG/EXR file
PNG/EXR file /tmp/coreai_out_{uuid}.png
    │
    ▼  HTTP response (just the path string)
    │
bridge.py reads file → ComfyUI tensor
```

**Why file-based:**
- ComfyUI already works with temp files (LoadImage, SaveImage)
- No shared memory complexity across Python↔Swift
- Debuggable: user can inspect /tmp/coreai_* files
- The bottleneck is inference (15ms-17s), not file I/O (~2ms for a 1024x1024 PNG)

---

## Distribution & Versioning

### Binary Distribution

```
GitHub Releases:
  v1.0.0/
    coreai-runner-arm64-macos     # ~15-20MB, stripped
    coreai-runner-arm64-macos.sig # SHA256 checksum
    checksums.txt                 # All checksums

Python package downloads on first run:
  comfy node install ComfyUI-CoreAI
    → pip installs package (no binary)
    → first predict() call downloads binary via GitHub Release
    → cached in package bin/ directory
```

### Versioning Scheme

```
ComfyUI-CoreAI v{NODE_VERSION}
  ├── Node Python code: semver (1.0.0, 1.1.0, ...)
  ├── Runner binary: pinned in node (runner v{RUNNER_VERSION})
  └── Catalog API: fetched at runtime (always latest)

node version → runner version mapping:
  ComfyUI-CoreAI 1.0.0 → coreai-runner 1.0.0
  ComfyUI-CoreAI 1.1.0 → coreai-runner 1.0.0 (no runner change)
  ComfyUI-CoreAI 1.2.0 → coreai-runner 1.1.0 (runner updated)
```

### CI/CD Pipeline

```
.github/workflows/build-runner.yml:
  trigger: tag v*.*.* on main
  runner: macos-15 (Apple Silicon GitHub Actions)
  steps:
    - swift build -c release
    - strip binary
    - upload to GitHub Release

.github/workflows/test-python.yml:
  trigger: PR + push to main
  steps:
    - pytest tests/
    - smoke test: install node, start runner, call /v1/health
```

---

## Security Considerations

| Concern | Mitigation |
|---------|-----------|
| **Unix socket access** | Socket at `/tmp/coreai-runner.sock` with 0600 permissions (owner only) |
| **Model download** | Only from catalog-verified HuggingFace URLs. SHA256 verified against catalog provenance. |
| **Arbitrary file read** | Runner only reads files passed as `image_path` — no directory traversal. Paths are validated to be in `/tmp/` or ComfyUI's output directory. |
| **Subprocess injection** | Binary path resolved internally, never from user input. Env var override (`COREAI_RUNNER_PATH`) is dev-only, documented as unsafe. |

---

## Performance Expectations

Based on catalog benchmarks (GPU compute unit, no device specified):

| Operation | Model | Latency | vs PyTorch/MPS (est.) |
|-----------|-------|---------|----------------------|
| Depth estimation | DA3 Small | **15ms** | ~2-3x faster, less VRAM |
| Object detection | RF-DETR Nano | **8.6ms** | ~3x faster |
| Segmentation | SAM 3 | **550ms** | Comparable, less VRAM |
| VLM decode | Qwen3-VL 2B | **191 tok/s** | ~1.5x faster pipelined |
| Image generation | FLUX.2 klein 4B | **17.4s** | Comparable |

**Memory advantage:** Core AI models use Neural Engine + GPU + CPU in parallel. PyTorch/MPS is GPU-only. This means:
- Depth Anything 3 Small: 55MB Core AI vs ~300MB PyTorch VRAM
- RF-DETR Nano: 108MB Core AI vs ~500MB PyTorch VRAM
- SAM 3: 1.7GB Core AI (shared) vs ~4GB PyTorch VRAM

**The real win for ComfyUI workflows:** running vision preprocessors on ANE while the GPU is busy with diffusion. True hardware parallelism, not possible with PyTorch alone.

---

## Failure Modes & Edge Cases

| Scenario | Behavior |
|----------|----------|
| Non-Apple-Silicon machine | `install.py` detects arch, refuses install with clear message |
| Runner crashes mid-inference | bridge.py detects dead socket, restarts runner, retries once |
| Model not installed | Node shows "Download" button in UI, triggers `/load` which downloads |
| Insufficient memory | Runner returns `MEMORY_INSUFFICIENT`, node suggests smaller model variant |
| `patch_required` models (Qwen3-VL) | Runner applies patches automatically; first load takes longer |
| `aot_required` models (SAM 3) | First load triggers AOT compile; node shows "Compiling..." status |
| Thermal throttle | Runner reports `thermal_state: "fair/serious"`, node warns user |
| Multiple models needed | Runner uses LRU cache; node marks workflow-critical models as sticky |
| ComfyUI restart | Runner process killed by atexit hook; socket cleaned up |

---

## Open Questions — RESOLVED

All 3 critical questions answered with primary sources (Zoo source code,
CoreAIKit implementation, Apple docs). See
[`docs/OPEN_QUESTIONS_RESOLVED.md`](docs/OPEN_QUESTIONS_RESOLVED.md) for full
detail with citations.

### Q1: FLUX.2 multi-component → One directory, auto-detected

`PipelineDescriptor.resolve(at: url, config: .auto)` reads `metadata.json` and
resolves components by convention name. Runner passes one path, zero manual
component juggling. macOS-only for v1 (exceeds iOS memory limit).

### Q2: Qwen3-VL patch → Already in CoreAIKit

The `patch_required` flag refers to `apps/coreai-pipelined-static-inputs.patch`
— a runtime engine patch baked into `john-rocky/coreai-models` v0.1.2-zoo,
which CoreAIKit already links as an SPM dependency. **Our runner must link
CoreAIKit** (not just the raw system framework) to get the patched engine.

VLM = two sub-bundles (decoder + vision tower) paired by `VLModelID`.

### Q3: Concurrent inference → Yes, via cross-unit parallelism

Multiple `InferenceFunction` instances run independently. The real win is
**cross-unit parallelism**: depth/detection on ANE while diffusion runs on GPU.
`GraphModel` accepts `.neuralEngine` / `.gpu` / `.cpu`. `ObjectDetector`
supports split deployment (backbone→ANE, head→GPU).

### Architecture changes from findings

1. **Runner links CoreAIKit** (SPM) — not standalone system framework
2. **Binary ~25-30MB** (was ~15-20MB) — includes CoreAILM patched engine
3. **Compute unit routing** — ANE-eligible nodes get `.neuralEngine` default

---

## Roadmap

### MVP (v1.0.0)
- [ ] `coreai-runner` Swift binary with Hummingbird HTTP server
- [ ] `/v1/health`, `/v1/models`, `/v1/predict` endpoints
- [ ] `CoreAIRunner` wrapper (generic)
- [ ] 3 nodes: Depth, Detection, VLM
- [ ] `bridge.py` with subprocess lifecycle
- [ ] `install.py` with binary auto-download
- [ ] Catalog API integration (model dropdowns)
- [ ] CI: build runner on macOS GitHub Actions
- [ ] README + ARCHITECTURE docs

### v1.1.0
- [ ] `CoreAIImageSegmenter` wrapper
- [ ] SAM 3 node with point/box/text prompts
- [ ] Image Generation node (FLUX.2)
- [ ] LRU model cache with sticky models
- [ ] Progress reporting for model downloads

### v1.2.0
- [ ] `CoreAIDiffusionPipeline` wrapper
- [ ] Z-Image Turbo support
- [ ] Instance Segmentation node (RF-DETR-Seg)
- [ ] CLIP embedding node
- [ ] Benchmark suite (Core AI vs PyTorch/MPS)

### Future
- [ ] iOS runner variant
- [ ] LAN discovery (Bonjour/mDNS)
- [ ] Model quantization selector (fp16/fp32/int8)
- [ ] Workflow templates (depth → ControlNet, SAM → inpaint)
