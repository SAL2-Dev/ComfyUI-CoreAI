"""
ComfyUI-CoreAI — Apple Core AI models (Neural Engine) as ComfyUI vision nodes.

This __init__.py registers the node classes so ComfyUI discovers them
automatically when the package is in the custom_nodes directory.

Nodes:
    CoreAIDepthEstimation  — monocular depth (Depth Anything 3) → ControlNet
    CoreAIObjectDetection  — object detection (RF-DETR, YOLOX) → bboxes
    CoreAIVisionLanguage   — VLM (Qwen3-VL) → image captioning / Q&A
"""

from __future__ import annotations

import logging

# Set up logging — visible in ComfyUI console
logging.getLogger("ComfyUI-CoreAI").setLevel(logging.INFO)

from .nodes.depth import CoreAIDepthEstimation
from .nodes.detection import CoreAIObjectDetection
from .nodes.vlm import CoreAIVisionLanguage
from .nodes.segmentation import CoreAISegmentation
from .nodes.image_gen import CoreAIImageGeneration
from .nodes.embedding import CoreAIImageTextSimilarity
from .nodes.instance_seg import CoreAIInstanceSegmentation
from .nodes.loader import CoreAIModelLoader, CoreAIHealthCheck

# --- ComfyUI node registration ---

NODE_CLASS_MAPPINGS = {
    # Vision — inference
    "CoreAIDepthEstimation": CoreAIDepthEstimation,
    "CoreAIObjectDetection": CoreAIObjectDetection,
    "CoreAIVisionLanguage": CoreAIVisionLanguage,
    "CoreAISegmentation": CoreAISegmentation,
    "CoreAIInstanceSegmentation": CoreAIInstanceSegmentation,
    # Analysis
    "CoreAIImageTextSimilarity": CoreAIImageTextSimilarity,
    # Generation
    "CoreAIImageGeneration": CoreAIImageGeneration,
    # Utils
    "CoreAIModelLoader": CoreAIModelLoader,
    "CoreAIHealthCheck": CoreAIHealthCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreAIDepthEstimation": "CoreAI Depth Estimation",
    "CoreAIObjectDetection": "CoreAI Object Detection",
    "CoreAIVisionLanguage": "CoreAI Vision-Language (VLM)",
    "CoreAISegmentation": "CoreAI Segmentation (SAM 3)",
    "CoreAIInstanceSegmentation": "CoreAI Instance Segmentation",
    "CoreAIImageTextSimilarity": "CoreAI CLIP Similarity",
    "CoreAIImageGeneration": "CoreAI Image Generation (FLUX.2)",
    "CoreAIModelLoader": "CoreAI Model Loader",
    "CoreAIHealthCheck": "CoreAI Health Check",
}

__version__ = "1.0.0-dev"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
