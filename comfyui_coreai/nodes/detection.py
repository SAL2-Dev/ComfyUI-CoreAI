"""
nodes/detection.py — CoreAI Object Detection node for ComfyUI.

Runs object detection (RF-DETR, YOLOX) via Core AI. Outputs bounding
boxes and scores, plus an annotated image with drawn boxes.

Models: RF-DETR Nano/Small/Medium/Large (108-122MB, fp32, ~8.6ms)
        YOLOX-S (8.97M params, fp32)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .. import catalog
from ..bridge import get_runner
from ..image_utils import tensor_to_png, load_output_image, cleanup_temp

logger = logging.getLogger("ComfyUI-CoreAI")

# Cache model list for dropdown
_DETECT_MODELS: list[str] | None = None


def _get_detect_models() -> list[str]:
    global _DETECT_MODELS
    if _DETECT_MODELS is None:
        _DETECT_MODELS = catalog.model_dropdown(capability="object-detection")
        if not _DETECT_MODELS:
            _DETECT_MODELS = [
                "rf-detr-nano", "rf-detr-small", "rf-detr-medium", "rf-detr-large",
                "yolox-s",
            ]
    return _DETECT_MODELS


def draw_boxes_on_image(tensor, detections: list[dict]) -> Any:
    """Draw bounding boxes on a ComfyUI image tensor [B,C,H,W]."""
    try:
        import torch
        from PIL import Image, ImageDraw
    except ImportError:
        return tensor

    arr = tensor.detach().cpu().float().numpy()
    batched = arr.ndim == 4
    if batched:
        arr = arr[0]

    # CHW → HWC
    hwc = np.transpose(arr, (1, 2, 0))
    h, w = hwc.shape[:2]
    hwc_uint8 = np.clip(hwc * 255, 0, 255).astype(np.uint8)

    # Only handle RGB
    if hwc_uint8.shape[2] == 4:
        hwc_uint8 = hwc_uint8[:, :, :3]

    img = Image.fromarray(hwc_uint8)
    draw = ImageDraw.Draw(img)

    # COCO color palette (20 distinct colors)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
        (128, 0, 255), (0, 128, 255), (200, 200, 0), (200, 0, 200),
    ]

    for i, det in enumerate(detections):
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        # bbox is normalized [x1, y1, x2, y2]
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{det.get('label', '?')} {det.get('score', 0):.2f}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 15)), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, max(0, y1 - 15)), label, fill=(255, 255, 255))

    # Back to tensor
    annotated = np.array(img, dtype=np.float32) / 255.0
    annotated = np.transpose(annotated, (2, 0, 1))
    if batched:
        annotated = np.expand_dims(annotated, axis=0)
    return torch.from_numpy(annotated)


class CoreAIObjectDetection:
    """
    Object detection using Apple Core AI models.

    RF-DETR models need no NMS (DETR family). YOLOX-S is a dense
    detector (obj·cls + per-class NMS, handled by CoreAIKit).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_detect_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    models,
                    {
                        "default": models[0] if models else "rf-detr-nano",
                    },
                ),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum confidence score for detections.",
                    },
                ),
                "compute_unit": (
                    ["auto", "gpu", "neuralEngine", "cpu"],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("annotated_image", "detections_json", "num_detections")
    FUNCTION = "detect_objects"
    CATEGORY = "CoreAI/Vision"

    def detect_objects(
        self,
        image,
        model: str,
        score_threshold: float = 0.5,
        compute_unit: str = "auto",
    ):
        """Run object detection."""
        input_path = tensor_to_png(image)

        try:
            runner = get_runner()
            result = runner.predict(
                model_id=model,
                image_path=input_path,
                score_threshold=score_threshold,
                compute_unit=compute_unit,
            )

            detections = result["output"].get("detections", [])
            num = len(detections)

            # Draw boxes on the original image
            annotated = draw_boxes_on_image(image, detections)

            # JSON-serialize detections for STRING output
            import json
            detections_json = json.dumps(detections, indent=2)

            timing = result.get("timing", {})
            ms = timing.get("total_ms", 0)
            logger.info(
                "CoreAI Detect [%s]: %d objects in %.1fms",
                model, num, ms,
            )

            return (annotated, detections_json, num)

        finally:
            cleanup_temp(input_path)
