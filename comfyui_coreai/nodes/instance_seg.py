"""
nodes/instance_seg.py — CoreAI Instance Segmentation node for ComfyUI.

Runs RF-DETR-Seg models for instance segmentation with per-instance masks.
Output: annotated image with colored mask overlays + composite mask.

Models: RF-DETR-Seg Nano/Small/Medium/Large/XLarge/2Xlarge (fp32, no NMS)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .. import catalog
from ..bridge import get_runner
from ..image_utils import tensor_to_png, cleanup_temp

logger = logging.getLogger("ComfyUI-CoreAI")

_ISEG_MODELS: list[str] | None = None


def _get_iseg_models() -> list[str]:
    global _ISEG_MODELS
    if _ISEG_MODELS is None:
        _ISEG_MODELS = catalog.model_dropdown(capability="instance-segmentation")
        if not _ISEG_MODELS:
            _ISEG_MODELS = [
                "rf-detr-seg-nano", "rf-detr-seg-small", "rf-detr-seg-medium",
                "rf-detr-seg-large", "rf-detr-seg-xlarge", "rf-detr-seg-2xlarge",
            ]
    return _ISEG_MODELS


class CoreAIInstanceSegmentation:
    """
    Instance Segmentation using Apple Core AI (RF-DETR-Seg).

    Detects objects and produces per-instance segmentation masks.
    DETR-family: no NMS needed. Masks are raw logits (foreground = logit > 0).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_iseg_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    models,
                    {"default": models[0] if models else "rf-detr-seg-nano"},
                ),
                "score_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("annotated_image", "detections_json")
    FUNCTION = "segment_instances"
    CATEGORY = "CoreAI/Vision"

    def segment_instances(
        self,
        image,
        model: str,
        score_threshold: float = 0.5,
    ):
        """Run instance segmentation."""
        input_path = tensor_to_png(image)

        try:
            runner = get_runner()
            result = runner.predict(
                model_id=model,
                image_path=input_path,
                score_threshold=score_threshold,
            )

            detections = result["output"].get("detections", [])
            num = len(detections)

            # Reuse the detection box-drawing for annotated image
            from .detection import draw_boxes_on_image
            annotated = draw_boxes_on_image(image, detections)

            import json
            detections_json = json.dumps(detections, indent=2)

            timing = result.get("timing", {})
            ms = timing.get("total_ms", 0)
            logger.info("CoreAI InstanceSeg [%s]: %d instances in %.1fms", model, num, ms)

            return (annotated, detections_json)
        finally:
            cleanup_temp(input_path)
