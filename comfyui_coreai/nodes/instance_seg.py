"""
nodes/instance_seg.py — CoreAI Instance Segmentation node for ComfyUI.

Runs RF-DETR-Seg models for instance segmentation with per-instance masks.
Output: annotated image with colored mask overlays + composite mask.

Models: RF-DETR-Seg Nano/Small/Medium/Large/XLarge/2Xlarge (fp32, no NMS)
"""

from __future__ import annotations

import logging
from typing import Any

from .. import catalog
from ..bridge import get_runner
from ..image_utils import tensor_to_png, cleanup_temp

logger = logging.getLogger("ComfyUI-CoreAI")

_ISEG_MODELS: list[str] | None = None
_MODELS_FETCHED_AT: float = 0
_MODELS_TTL: float = 300  # 5-minute cache — catalog refresh interval


def _get_iseg_models() -> list[str]:
    import time
    global _ISEG_MODELS, _MODELS_FETCHED_AT
    if _ISEG_MODELS is None or time.monotonic() - _MODELS_FETCHED_AT > _MODELS_TTL:
        _ISEG_MODELS = catalog.model_dropdown(capability="instance-segmentation")
        if not _ISEG_MODELS:
            _ISEG_MODELS = ["official-yolox-instance-seg"]
        _MODELS_FETCHED_AT = time.monotonic()
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
    CATEGORY = "SAL2/Vision"

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
            try:
                result = runner.predict(
                        model_id=model,
                    image_path=input_path,
                    score_threshold=score_threshold,
            )
            except Exception as e:
                error_str = str(e).lower()
                if "framework" in error_str or "absent in this sdk" in error_str:
                    raise RuntimeError(
                        f"'{model}' requires a framework not available in this macOS/SDK version. "
                        f"This capability is supported on future macOS releases."
                    )
                if "not_installed" in error_str or "model_load_failed" in error_str:
                    raise RuntimeError(
                        f"Model '{model}' is not downloaded yet. "
                        f"Use the Download button on this node or the "
                        f"CoreAI Model Loader to install it first."
                    )
                raise

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
