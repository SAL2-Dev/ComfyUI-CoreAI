"""
ComfyUI-CoreAI — Apple Core AI models (Neural Engine) as ComfyUI vision nodes.

This __init__.py registers the node classes so ComfyUI discovers them
automatically when the package is in the custom_nodes directory.

Only nodes whose required system frameworks are available in the current
macOS/SDK are registered. Segmentation (SAM) and Image Generation (FLUX.2)
require CoreAIImageSegmenter / CoreAIDiffusionPipeline, which ship in future
macOS releases. They will be auto-enabled when the frameworks become available.
"""

from __future__ import annotations

import logging
import platform

# Set up logging — visible in ComfyUI console
logging.getLogger("ComfyUI-CoreAI").setLevel(logging.INFO)
logger = logging.getLogger("ComfyUI-CoreAI")

from .nodes.depth import CoreAIDepthEstimation
from .nodes.detection import CoreAIObjectDetection
from .nodes.vlm import CoreAIVisionLanguage
from .nodes.segmentation import CoreAISegmentation
from .nodes.image_gen import CoreAIImageGeneration
from .nodes.embedding import CoreAIImageTextSimilarity
from .nodes.instance_seg import CoreAIInstanceSegmentation
from .nodes.loader import CoreAIModelLoader, CoreAIHealthCheck
from .nodes.apple_text import CoreAIAppleText

# --- Capability detection --------------------------------------------------

def _macos_major() -> int:
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except Exception:
        return 0

_MACOS_MAJOR = _macos_major()

# Core AI vision nodes require macOS 27+ (Core AI framework)
_COREAI_AVAILABLE = _MACOS_MAJOR >= 27

# Apple Text requires macOS 26+ (FoundationModels)
_FOUNDATION_MODELS_AVAILABLE = _MACOS_MAJOR >= 26

# SAM / FLUX.2 require CoreAIImageSegmenter / CoreAIDiffusionPipeline
# (not yet in macOS 27 SDK — will auto-work when the frameworks ship)
_SEGMENTATION_AVAILABLE = False
_IMAGE_GEN_AVAILABLE = False

# --- ComfyUI node registration ---

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

if _FOUNDATION_MODELS_AVAILABLE:
    NODE_CLASS_MAPPINGS["CoreAIAppleText"] = CoreAIAppleText
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIAppleText"] = "Apple Text (On-Device)"

if _COREAI_AVAILABLE:
    # Vision — inference (Core AI / Neural Engine)
    NODE_CLASS_MAPPINGS["CoreAIDepthEstimation"] = CoreAIDepthEstimation
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIDepthEstimation"] = "Depth Estimation"

    NODE_CLASS_MAPPINGS["CoreAIObjectDetection"] = CoreAIObjectDetection
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIObjectDetection"] = "Object Detection"

    NODE_CLASS_MAPPINGS["CoreAIInstanceSegmentation"] = CoreAIInstanceSegmentation
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIInstanceSegmentation"] = "Instance Segmentation"

    NODE_CLASS_MAPPINGS["CoreAIVisionLanguage"] = CoreAIVisionLanguage
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIVisionLanguage"] = "Vision-Language (VLM)"

    # Analysis
    NODE_CLASS_MAPPINGS["CoreAIImageTextSimilarity"] = CoreAIImageTextSimilarity
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIImageTextSimilarity"] = "CLIP Similarity"

    # Utils
    NODE_CLASS_MAPPINGS["CoreAIModelLoader"] = CoreAIModelLoader
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIModelLoader"] = "Model Loader"

    NODE_CLASS_MAPPINGS["CoreAIHealthCheck"] = CoreAIHealthCheck
    NODE_DISPLAY_NAME_MAPPINGS["CoreAIHealthCheck"] = "Health Check"

    # Registered but not yet runnable — keeps saved workflows valid
    if not _SEGMENTATION_AVAILABLE:
        NODE_CLASS_MAPPINGS["CoreAISegmentation"] = CoreAISegmentation
        NODE_DISPLAY_NAME_MAPPINGS["CoreAISegmentation"] = "Segmentation (SAM) — coming soon"

    if not _IMAGE_GEN_AVAILABLE:
        NODE_CLASS_MAPPINGS["CoreAIImageGeneration"] = CoreAIImageGeneration
        NODE_DISPLAY_NAME_MAPPINGS["CoreAIImageGeneration"] = "Image Generation (FLUX.2) — coming soon"
else:
    logger.warning(
        "CoreAI vision nodes require macOS 27+. "
        "Only Apple Text is available on macOS %s.",
        platform.mac_ver()[0],
    )

__version__ = "1.0.0-dev"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Web extension — adds status badge + download button to CoreAI nodes
WEB_DIRECTORY = "./web"

# Register HTTP API routes on ComfyUI server startup
try:
    from .api import register_routes
    register_routes()
except ImportError:
    pass  # Running outside ComfyUI (e.g., in tests)
except Exception:
    pass  # PromptServer not yet initialized
