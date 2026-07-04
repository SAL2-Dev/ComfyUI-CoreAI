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

# --- ComfyUI node registration ---

NODE_CLASS_MAPPINGS = {
    "CoreAIDepthEstimation": CoreAIDepthEstimation,
    "CoreAIObjectDetection": CoreAIObjectDetection,
    "CoreAIVisionLanguage": CoreAIVisionLanguage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreAIDepthEstimation": "CoreAI Depth Estimation",
    "CoreAIObjectDetection": "CoreAI Object Detection",
    "CoreAIVisionLanguage": "CoreAI Vision-Language (VLM)",
}

__version__ = "1.0.0-dev"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
