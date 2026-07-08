"""
nodes/embedding.py — CoreAI CLIP Image-Text Similarity node for ComfyUI.

Computes CLIP ViT-B/32 embeddings for images and/or text. Useful for
prompt scoring, image retrieval, or evaluating diffusion outputs.

Models: CLIP ViT-B/32 (151M, fp16, MIT license)
"""

from __future__ import annotations

import logging
from typing import Any

from .. import catalog
from ..bridge import get_runner
from ..image_utils import tensor_to_png, cleanup_temp
from ..perf import format_vision_perf, with_perf

logger = logging.getLogger("ComfyUI-CoreAI")

_EMBED_MODELS: list[str] | None = None
_MODELS_FETCHED_AT: float = 0
_MODELS_TTL: float = 300  # 5-minute cache — catalog refresh interval


def _get_embed_models() -> list[str]:
    import time
    global _EMBED_MODELS, _MODELS_FETCHED_AT
    if _EMBED_MODELS is None or time.monotonic() - _MODELS_FETCHED_AT > _MODELS_TTL:
        _EMBED_MODELS = catalog.model_dropdown(capability="image-text-similarity")
        if not _EMBED_MODELS:
            _EMBED_MODELS = ["official-clip-vit-base-patch32"]
        _MODELS_FETCHED_AT = time.monotonic()
    return _EMBED_MODELS


class CoreAIImageTextSimilarity:
    """
    CLIP Image-Text Similarity using Apple Core AI.

    Computes cosine similarity between an image and one or more text
    captions. Useful for prompt scoring or image retrieval.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_embed_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "captions": (
                    "STRING",
                    {
                        "default": "a photo of a cat\na photo of a dog\na landscape",
                        "multiline": True,
                        "tooltip": "One caption per line.",
                    },
                ),
                "model": (
                    models,
                    {"default": models[0] if models else "official-clip-vit-base-patch32"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scores",)
    FUNCTION = "compute_similarity"
    CATEGORY = "CoreAI/Analysis"

    def compute_similarity(self, image, captions: str, model: str):
        input_path = tensor_to_png(image)
        caption_list = [c.strip() for c in captions.split("\n") if c.strip()]

        try:
            runner = get_runner()
            try:
                result = runner.predict(
                        model_id=model,
                    image_path=input_path,
                    prompt="|||".join(caption_list),
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

            timing = result.get("timing", {})
            perf = format_vision_perf(timing)

            text_output = result["output"].get("text", "")
            if text_output:
                try:
                    import json
                    scores = json.loads(text_output)
                    if isinstance(scores, list):
                        lines = [f"{s:.4f}  {c}" for s, c in zip(scores, caption_list)]
                        return with_perf(("\n".join(lines),), perf)
                except (json.JSONDecodeError, TypeError):
                    pass

            return with_perf((text_output or "No scores returned",), perf)
        finally:
            cleanup_temp(input_path)
