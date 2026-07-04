# Core AI Open Questions — Resolved

> Answers to the 3 critical architecture questions, backed by Apple docs,
> Zoo source code, and CoreAIKit implementation. All citations are to
> primary sources (no assumptions).

---

## Q1: FLUX.2 Multi-Component Loading

**Question:** The FLUX.2 model has 4 separate `.aimodel` packages (TextEncoder,
Transformer, VAEDecoder, VAEEncoder). Does `CoreAIDiffusionPipeline` handle this
as one path, or do we pass each component?

### Answer: One bundle directory. Auto-detected by name.

The pipeline takes **one directory path** containing all components. The type
(FLUX.2 / SD3 / SD) is auto-detected from `metadata.json`, and each component
is resolved by **convention name** at load time.

**Source:** `apps/CoreAIImageGen/Sources/DiffusionEngine.swift` (Zoo)

```swift
// One path → auto-detection
let desc = try PipelineDescriptor.resolve(at: url, config: .auto)

// PipelineDescriptor reads metadata.json → determines type
switch desc.type {
case .some(.flux2):
    built = try await Flux2Pipeline(from: url, config: .auto, mode: fluxMode)
case .some(.stableDiffusion3):
    built = try await SD3Pipeline(from: url, config: .auto)
default:
    built = try await StableDiffusionPipeline.load(from: url, config: .auto)
}
```

**Directory layout (macOS, full resolution):**
```
FLUX.2-klein-4B/
├── metadata.json          ← pipeline type + config
├── TextEncoder.aimodel/   ← resolved by name "TextEncoder"
├── Transformer.aimodel/   ← resolved by name "Transformer"
├── VAEDecoder.aimodel/    ← resolved by name "VAEDecoder"
├── VAEEncoder.aimodel/    ← resolved by name "VAEEncoder"
├── tokenizer/
├── vae_bn_mean.npy
└── vae_bn_var.npy
```

**iOS half-resolution variant:** the HF bundle is universal — the platform
fetches only the subset it needs:

| Platform | Components fetched |
|----------|-------------------|
| macOS    | `Transformer.aimodel`, `TextEncoder.aimodel`, `VAEDecoder.aimodel`, `VAEEncoder.aimodel` |
| iOS      | `Transformer_512.aimodel`, `TextEncoder.aimodel`, `VAEDecoder_half.aimodel`, `VAEEncoder_half.aimodel` |

**Our runner API stays simple:**
```swift
// Runner receives one path, pipeline auto-resolves components
let pipeline = try await Flux2Pipeline(from: bundleDirURL, config: .auto, mode: .auto)
let result = try await pipeline.generateImages(configuration: config) { progress in ... }
```

**Additional findings:**
- `vae_bn_mean.npy` and `vae_bn_var.npy` are tiny root-level files (640 bytes
  each) that the HF tree API can't enumerate (only lists directories). They
  must be fetched with direct resolve GETs.
- Scheduler is `discreteFlow` for FLUX.2 (not dpmSolverMultistep).
- `lazyModelLoading: true` — the text encoder is loaded on demand and released
  before the transformer runs to manage memory.
- 4-step distilled, guidance 1.0.
- **macOS only for our v1** — at 4B params the peak footprint exceeds the iOS
  per-process memory limit even AOT-compiled.

### Impact on our architecture

The runner's `DiffusionRunner` receives one `bundle_dir_path` and delegates to
`PipelineDescriptor.resolve()` — zero manual component juggling. The download
step fetches the whole repo subtree (same as any other model).

---

## Q2: Qwen3-VL `patch_required: true`

**Question:** What does `patch_required` mean? Are patches bundled in the
`.aimodel` package, or fetched separately?

### Answer: It's an engine-level patch, NOT bundled with the model.

The "patch" is **not** a per-model file inside the `.aimodel` package. It is a
**runtime engine patch** applied to the pipelined engine itself, bundled in the
Zoo's `apps/` directory as a `.patch` file.

**Source:** `zoo/qwen3-vl.md` (Zoo)

> The whole multimodal state rides the **static-input hook**
> (`apps/coreai-pipelined-static-inputs.patch`) plus an id-space trick —
> the graph stays `ids + positions → logits`.

**The patch:** `apps/coreai-pipelined-static-inputs.patch` modifies the
pipelined engine (from `apple/coreai-models`) to add a **static-input hook** —
named MTLBuffers that the host binds on every encode. This is how image
embeddings get injected into the text decoder without the graph knowing about
images.

**How it's consumed:**

```
Apple coreai-models (upstream)
    │
    ▼  + apps/coreai-pipelined-static-inputs.patch
    │
john-rocky/coreai-models (patched fork, tagged v0.1.2-zoo)
    │
    ▼  imported as SPM dependency
    │
CoreAIKit (links CoreAILM product from patched fork)
    │
    ▼  VLRuntime uses the patched engine
    │
Qwen3-VL works
```

**Critical detail:** The patched fork is already a **compiled SPM dependency**
in CoreAIKit's `Package.swift`:

```swift
.package(url: "https://github.com/john-rocky/coreai-models", exact: "0.1.2-zoo"),
```

This means: **if our runner links CoreAIKit (or its CoreAILM product), the
patch is already baked in.** We don't apply it ourselves — it's already in
the compiled library.

**The Qwen3-VL architecture (from the card):**
- Decoder = pure Qwen3 text LLM, **unmodified** — same bundle works as text-only
- Vision tower = separate `.aimodel`, runs the ViT encoder once per image
- Host writes `image_embeds` + `deepstack_embeds` into MTLBuffers
- Prompt `<|image_pad|>` ids are rewritten to `V + slot` (extension ids)
- M-RoPE is derived entirely in-graph from (ids, position) — no extra state
- KV cache is the engine's native pair (pure attention, no extra states)

**Two sub-bundles per VLM:**

```
Qwen3-VL-2B-CoreAI/
└── gpu-pipelined/
    ├── qwen3_vl_2b_instruct_decode_int8hu_s1/   ← decoder (text LLM)
    │   ├── metadata.json
    │   ├── *.aimodel/
    │   └── tokenizer/
    └── qwen3_vl_2b_instruct_vision/             ← vision tower (ViT)
        └── *.aimodel/
```

CoreAIKit's `VLModelID` pairs them:
```swift
public static let qwen3VL2B = VLModelID(
    decoder: ModelID("mlboydaisuke/Qwen3-VL-2B-CoreAI",
        path: "gpu-pipelined/qwen3_vl_2b_instruct_decode_int8hu_s1"),
    vision: ModelID("mlboydaisuke/Qwen3-VL-2B-CoreAI",
        path: "gpu-pipelined/qwen3_vl_2b_instruct_vision"),
    arch: .qwen3VL2B)
```

**With zero embeds and `start = 1<<30` the decoder IS a plain Qwen3 text LLM.**

### Impact on our architecture

**`patch_required` is a non-issue for our runner IF we link CoreAIKit as an SPM
dependency.** The patched engine ships inside CoreAIKit's compiled binary.

**However**, this means our `coreai-runner` binary **must link CoreAIKit** (or
at minimum `CoreAILM` + `CoreAIKitVision`), not just the raw system `CoreAI`
framework. The dependency chain:

```
coreai-runner
  └─ CoreAIKit (SPM)
       ├─ CoreAIKitVision (system CoreAI framework — GraphModel, DepthEstimator, ObjectDetector)
       ├─ CoreAILM (from john-rocky/coreai-models v0.1.2-zoo — patched pipelined engine)
       └─ swift-transformers (HF tokenizer)
```

**This changes the binary from "standalone Swift + system Core AI" to
"CoreAIKit SPM consumer."** The binary grows from ~15MB to ~25-30MB (CoreAIKit
+ CoreAILM + swift-transformers), but gains all model types for free, including
patched models, VLMs, and the pipelined engine.

---

## Q3: Concurrent Neural Engine Sessions

**Question:** Can the runner handle 2 simultaneous inference calls on different
models? Does Foundation Models framework allow concurrent sessions?

### Answer: Yes — but with a critical hardware constraint.

The Foundation Models framework / Core AI runtime supports multiple concurrent
`InferenceFunction` instances. Each `.aimodel` loaded via `AIModel(contentsOf:)`
gets its own compiled function and can run independently.

**However**, there is a physical constraint on the **Neural Engine (ANE)**:

### Compute Unit Topology on Apple Silicon

```
┌──────────────────────────────────────────────────┐
│              Apple Silicon Chip                   │
│                                                   │
│  ┌─────────┐  ┌─────────┐  ┌──────────────────┐ │
│  │   CPU   │  │   GPU   │  │ Neural Engine    │ │
│  │ (P+E)   │  │         │  │ (ANE)            │ │
│  └────┬────┘  └────┬────┘  └────────┬─────────┘ │
│       │            │                │            │
│       └────────────┴────────────────┘            │
│                    │                              │
│              Unified Memory                       │
└──────────────────────────────────────────────────┘
```

- **GPU**: Multiple models CAN run concurrently, but they time-share the GPU
  cores (context switching overhead).
- **ANE**: The ANE is a **dedicated coprocessor** that operates independently
  from the GPU. It has its own scheduling queue.
- **CPU**: Can always do work alongside both.

**The key insight for ComfyUI workflows:**

The real win is **cross-unit parallelism**, not same-unit concurrency:

```
WORKFLOW EXAMPLE (true parallelism):

  Step 1 (GPU): FLUX.2 generating image          ← 17.4s
  Step 2 (ANE): Depth estimation on prev image    ← 15ms  ← SIMULTANEOUS
  Step 3 (ANE): Object detection on prev image    ← 8.6ms ← SIMULTANEOUS
```

While the GPU is saturated with diffusion, the ANE processes vision tasks
in parallel — zero GPU contention.

### How CoreAIKit handles this

CoreAIKit's `GraphModel` takes a `computeUnits` parameter:

```swift
let depthModel = try await GraphModel(
    contentsOf: url, computeUnits: .neuralEngine)  // ANE
let detectModel = try await GraphModel(
    contentsOf: url, computeUnits: .gpu)            // GPU
```

And `ObjectDetector` supports **split deployment** — backbone on ANE, head on GPU:

```swift
// RF-DETR split: ViT backbone → ANE, deformable head → GPU
let detector = try await ObjectDetector(
    backboneAt: backboneURL, headAt: headURL,
    backboneUnits: .neuralEngine,
    headUnits: .gpu)
```

### Our concurrency model

```swift
// The runner uses Swift's async/await — requests are inherently concurrent
// The HTTP server (Hummingbird) handles multiple connections in parallel

// Concurrency is safe BECAUSE:
// 1. Each InferenceFunction is an independent compiled artifact
// 2. Swift actors enforce thread safety on shared model state
// 3. The ANE and GPU operate as independent processors

func predict(req: PredictRequest) async throws -> PredictResponse {
    // This call does NOT block other in-flight predictions on different models
    let result = try await modelCache.get(req.model_id).predict(req.input)
    return result
}
```

**Recommendation for our runner:**

| Strategy | Implementation |
|----------|---------------|
| Default compute unit | `.gpu` (matches catalog benchmarks) |
| Depth/Detection models | Offer `.neuralEngine` as option (frees GPU) |
| Split models (RF-DETR) | Auto-detect split bundles, offer backbone→ANE |
| Concurrency | HTTP server handles parallel requests natively |
| Model cache | Actor-isolated, LRU eviction on memory pressure |

**The ComfyUI workflow advantage:**

In ComfyUI, nodes are evaluated as a DAG. If two nodes don't depend on each
other, ComfyUI can execute them concurrently. With our runner:

```
[Load Image] ──┬── [CoreAI Depth (ANE)]  ──┐
               │                            ├── [Apply Depth ControlNet]
               └── [CoreAI Detect (GPU)] ──┘
                    ↑ runs in PARALLEL with Depth (different compute unit)
```

This is impossible with PyTorch/MPS — everything competes for the same GPU.

---

## Consolidated Impact on Architecture

These findings change three things in ARCHITECTURE.md:

### Change 1: Runner links CoreAIKit (not raw system framework)

```
BEFORE: coreai-runner → system CoreAI framework (standalone)
AFTER:  coreai-runner → CoreAIKit (SPM) → CoreAILM (patched engine) + CoreAIKitVision
```

**Benefits:**
- All model types supported (vision, VLM, LLM, diffusion, ASR)
- Patched engine baked in (Qwen3-VL works without manual patches)
- Proven model loading, download, caching (ModelStore)
- Typed pipelines: DepthEstimator, ObjectDetector, KitVisionModel, etc.

**Trade-off:**
- Binary size: ~25-30MB (was ~15-20MB)
- Build requires Swift 6.0 + macOS 27 SDK (for CoreAIKit platforms: `.macOS("27.0")`)

### Change 2: FLUX.2 node uses PipelineDescriptor, not manual components

```swift
// BEFORE (assumed): manually load 4 components
// AFTER (confirmed): one path, auto-detection
let desc = try PipelineDescriptor.resolve(at: bundleDirURL, config: .auto)
```

### Change 3: Compute unit routing enables true ComfyUI parallelism

```
Nodes that should prefer ANE:
  - Depth estimation (small, fast, frees GPU for diffusion)
  - RF-DETR backbone (ViT → ANE, deformable head → GPU)
  - CLIP image encoder

Nodes that stay on GPU (default):
  - SAM 3 (1.7GB, complex graph)
  - RF-DETR full (single graph)
  - FLUX.2 (diffusion pipeline)
```

---

## Sources

| Claim | Source |
|-------|--------|
| FLUX.2 uses `PipelineDescriptor.resolve()` + convention names | `apps/CoreAIImageGen/Sources/DiffusionEngine.swift` (Zoo) |
| FLUX.2 is 4-step, guidance 1.0, discreteFlow scheduler | `official/README.md` (Zoo) |
| FLUX.2 is macOS-only (exceeds iOS memory) | `official/README.md` (Zoo) |
| Qwen3-VL patch is `apps/coreai-pipelined-static-inputs.patch` | `zoo/qwen3-vl.md` (Zoo) |
| Patched engine imported as SPM `0.1.2-zoo` | `Package.swift` (CoreAIKit) |
| VLModelID pairs decoder + vision bundles | `Sources/CoreAIKit/Vision/KitVisionModel.swift` (CoreAIKit) |
| `GraphModel` supports `.neuralEngine` / `.gpu` / `.cpu` | `Sources/CoreAIKitVision/GraphModel.swift` (CoreAIKit) |
| `ObjectDetector` supports split ANE+GPU deployment | `Sources/CoreAIKitVision/ObjectDetector.swift` (CoreAIKit) |
| `DepthEstimator` folds ImageNet normalization in-graph | `Sources/CoreAIKitVision/DepthEstimator.swift` (CoreAIKit) |
| `ModelStore` does atomic download + staging | `Sources/CoreAIKitCore/ModelStore.swift` (CoreAIKit) |
| Foundation Models framework: iOS 26+, macOS 26+ | [Apple Developer Documentation](https://developer.apple.com/documentation/foundationmodels) |
