"""
nodes/segmentation.py — CoreAI Promptable Segmentation node for ComfyUI.

Runs SAM 3 (text-prompt, open-vocabulary) via Core AI. Give it an image
and a phrase ("cat", "the red car") → instance masks + boxes + scores.

Output: composite mask image (white on black) + individual mask tensors.

Models: SAM 3 (1.7GB, fp16, ~550ms), EfficientSAM3-TinyViT
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .. import catalog
from ..bridge import get_runner
from ..image_utils import tensor_to_png, load_output_image, cleanup_temp

logger = logging.getLogger("ComfyUI-CoreAI")

_SEG_MODELS: list[str] | None = None
_MODELS_FETCHED_AT: float = 0
_MODELS_TTL: float = 300  # 5-minute cache — catalog refresh interval


def _get_seg_models() -> list[str]:
    import time
    global _SEG_MODELS, _MODELS_FETCHED_AT
    if _SEG_MODELS is None or time.monotonic() - _MODELS_FETCHED_AT > _MODELS_TTL:
        _SEG_MODELS = catalog.model_dropdown(capability="promptable-segmentation")
        if not _SEG_MODELS:
            _SEG_MODELS = ["official-sam-3", "efficientsam3-tinyvit"]
        _MODELS_FETCHED_AT = time.monotonic()
    return _SEG_MODELS


def composite_masks(masks: list[dict], width: int, height: int):
    """Composite multiple binary mask PNGs into a single mask tensor."""
    try:
        import torch
    except ImportError:
        return None

    composite = np.zeros((height, width), dtype=np.float32)

    for mask_info in masks:
        mask_path = mask_info.get("mask_path")
        if not mask_path:
            continue
        try:
            mask_tensor = load_output_image(mask_path)
            mask_arr = mask_tensor.detach().cpu().numpy()[0]
            if mask_arr.shape[0] >= 1:
                mask_arr = mask_arr[0]
            if mask_arr.shape != (height, width):
                from PIL import Image
                pil = Image.fromarray(np.clip(mask_arr * 255, 0, 255).astype(np.uint8))
                pil = pil.resize((width, height), Image.NEAREST)
                mask_arr = np.array(pil, dtype=np.float32) / 255.0
            composite = np.maximum(composite, mask_arr)
            cleanup_temp(mask_path)
        except Exception as e:
            logger.warning("Failed to load mask %s: %s", mask_path, e)

    rgb = np.stack([composite, composite, composite], axis=0)
    rgb = np.expand_dims(rgb, axis=0)
    return torch.from_numpy(rgb)


class CoreAISegmentation:
    """
    Promptable Segmentation using Apple Core AI (SAM 3).

    Text-prompt, open-vocabulary: give it a phrase like "cat" or
    "the red car" and it returns instance masks with bounding boxes
    and confidence scores.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_seg_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    models,
                    {"default": models[0] if models else "official-sam-3"},
                ),
                "text_prompt": (
                    "STRING",
                    {
                        "default": "cat",
                        "multiline": False,
                        "tooltip": "What to segment. Open-vocabulary.",
                    },
                ),
                "score_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("mask_overlay", "composite_mask", "segment_info")
    FUNCTION = "segment"
    CATEGORY = "CoreAI/Vision"

    def segment(self, image, model: str, text_prompt: str, score_threshold: float = 0.5):
        input_path = tensor_to_png(image)

        try:
            img_np = image.detach().cpu().numpy()
            if img_np.ndim == 4:
                img_np = img_np[0]
            h, w = img_np.shape[-2], img_np.shape[-1]
        except Exception:
            h, w = 512, 512

        try:
            runner = get_runner()
            result = runner.predict(
                model_id=model,
                image_path=input_path,
                text_prompt=text_prompt,
                score_threshold=score_threshold,
            )

            masks = result["output"].get("mask_paths", [])
            filtered = [m for m in masks if m.get("score", 0) >= score_threshold]
            composite = composite_masks(filtered, w, h)

            import json
            segment_info = json.dumps(filtered, indent=2)

            timing = result.get("timing", {})
            ms = timing.get("total_ms", 0)
            logger.info("CoreAI Segmentation [%s]: %d segments in %.1fms", model, len(filtered), ms)

            return (composite, composite, segment_info)
        finally:
            cleanup_temp(input_path)
