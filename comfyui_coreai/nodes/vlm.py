"""
nodes/vlm.py — CoreAI Vision-Language Model node for ComfyUI.

Runs a VLM (Qwen3-VL) to describe, caption, or answer questions about
an image. Output is text — ideal for auto-generating prompts for
Stable Diffusion / FLUX.

Models: Qwen3-VL 2B (2.3GB, int8, ~191 tok/s)
        Qwen3-VL 4B (4.7GB, int8)
        MiniCPM-V 4.6 (sub-2B)
"""

from __future__ import annotations

import logging
from typing import Any

from .. import catalog
from ..bridge import get_runner
from ..image_utils import tensor_to_png, cleanup_temp

logger = logging.getLogger("ComfyUI-CoreAI")

# Cache model list
_VLM_MODELS: list[str] | None = None
_MODELS_FETCHED_AT: float = 0
_MODELS_TTL: float = 300  # 5-minute cache — catalog refresh interval


def _get_vlm_models() -> list[str]:
    import time
    global _VLM_MODELS, _MODELS_FETCHED_AT
    if _VLM_MODELS is None or time.monotonic() - _MODELS_FETCHED_AT > _MODELS_TTL:
        _VLM_MODELS = catalog.model_dropdown(capability="vision-language")
        if not _VLM_MODELS:
            _VLM_MODELS = ["qwen3-vl-2b", "minicpm-v-4-6"]
        _MODELS_FETCHED_AT = time.monotonic()
    return _VLM_MODELS


class CoreAIVisionLanguage:
    """
    Vision-Language Model using Apple Core AI.

    Give it an image and a text prompt → get a text response.
    The model runs entirely on-device via the pipelined engine with
    the static-inputs patch (baked into CoreAIKit).

    Common use: caption an image → feed to a diffusion prompt.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_vlm_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    models,
                    {
                        "default": models[0] if models else "qwen3-vl-2b",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe this image in detail.",
                        "multiline": True,
                        "tooltip": "Instruction or question about the image.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 200,
                        "min": 1,
                        "max": 4096,
                        "step": 10,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Higher = more creative, lower = more deterministic.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run_vlm"
    CATEGORY = "CoreAI/Vision"

    def run_vlm(
        self,
        image,
        model: str,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 200,
        temperature: float = 0.7,
    ):
        """Run the VLM and return generated text."""
        input_path = tensor_to_png(image)

        try:
            runner = get_runner()
            try:
                result = runner.predict(
                        model_id=model,
                    image_path=input_path,
                    prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
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

            text = result["output"].get("text", "")

            timing = result.get("timing", {})
            ms = timing.get("total_ms", 0)
            tokens = len(text.split())
            logger.info(
                "CoreAI VLM [%s]: %d words in %.1fms (%.0f words/s)",
                model, tokens, ms, (tokens / ms * 1000) if ms > 0 else 0,
            )

            return (text,)

        finally:
            cleanup_temp(input_path)
