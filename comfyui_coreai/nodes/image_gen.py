"""
nodes/image_gen.py — CoreAI Image Generation node for ComfyUI.

Generates images from text prompts using FLUX.2 klein 4B via Core AI's
CoreAIDiffusionPipeline runtime.

Models: FLUX.2 klein 4B (4GB, int4, ~17.4s for 1024px @ 4 steps)
"""

from __future__ import annotations

import logging
from typing import Any

from .. import catalog
from ..bridge import get_runner
from ..image_utils import load_output_image, cleanup_temp

logger = logging.getLogger("ComfyUI-CoreAI")

_GEN_MODELS: list[str] | None = None
_MODELS_FETCHED_AT: float = 0
_MODELS_TTL: float = 300  # 5-minute cache — catalog refresh interval


def _get_gen_models() -> list[str]:
    import time
    global _GEN_MODELS, _MODELS_FETCHED_AT
    if _GEN_MODELS is None or time.monotonic() - _MODELS_FETCHED_AT > _MODELS_TTL:
        _GEN_MODELS = catalog.model_dropdown(capability="image-generation")
        if not _GEN_MODELS:
            _GEN_MODELS = ["official-flux-2-klein-4b", "z-image-turbo"]
        _MODELS_FETCHED_AT = time.monotonic()
    return _GEN_MODELS


class CoreAIImageGeneration:
    """
    Image Generation using Apple Core AI (FLUX.2 klein 4B).

    Text-to-image via the stock CoreAIDiffusionPipeline runtime.
    4-step distilled, guidance 1.0, discreteFlow scheduler.
    macOS-only (exceeds iOS memory at 4B params).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_gen_models()
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "a serene mountain landscape at golden hour",
                        "multiline": True,
                    },
                ),
                "model": (
                    models,
                    {"default": models[0] if models else "official-flux-2-klein-4b"},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**32 - 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "CoreAI/Generation"

    def generate(self, prompt: str, model: str, seed: int = 0):
        try:
            runner = get_runner()
            result = runner.predict(model_id=model, prompt=prompt)

            output_path = result["output"].get("output_path")
            if not output_path:
                raise RuntimeError(f"Runner returned no output_path. Response: {result}")

            image_tensor = load_output_image(output_path)
            cleanup_temp(output_path)

            timing = result.get("timing", {})
            ms = timing.get("total_ms", 0)
            logger.info("CoreAI ImageGen [%s]: %.1fms (%.1fs)", model, ms, ms / 1000)

            return (image_tensor,)
        except Exception as e:
            logger.error("Image generation failed: %s", e)
            raise
