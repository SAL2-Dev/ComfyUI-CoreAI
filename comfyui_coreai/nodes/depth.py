"""
nodes/depth.py — CoreAI Depth Estimation node for ComfyUI.

Runs monocular depth estimation (Depth Anything 3) via the Core AI
Neural Engine / GPU. Output is a grayscale depth map usable as a
ControlNet preprocessor.

Models: Depth Anything 3 Small (54.5MB, fp16, ~15ms)
        Depth Anything 3 Base (202MB, fp16)
"""

from __future__ import annotations

import logging
from typing import Any

from .. import catalog
from ..bridge import get_runner
from ..image_utils import tensor_to_png, load_output_image, cleanup_temp

logger = logging.getLogger("ComfyUI-CoreAI")

# Cache model list for dropdown — refreshed when the catalog cache expires
_DEPTH_MODELS: list[str] | None = None


def _get_depth_models() -> list[str]:
    global _DEPTH_MODELS
    if _DEPTH_MODELS is None:
        _DEPTH_MODELS = catalog.model_dropdown(capability="monocular-depth")
        if not _DEPTH_MODELS:
            # Fallback if catalog is unreachable
            _DEPTH_MODELS = ["depth-anything-3-small", "depth-anything-3-base"]
    return _DEPTH_MODELS


class CoreAIDepthEstimation:
    """
    Monocular depth estimation using Apple Core AI models.

    Runs on Neural Engine (default) or GPU. The depth map output can be
    fed directly into a ControlNet (Depth) node.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_depth_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    models,
                    {
                        "default": models[0] if models else "depth-anything-3-small",
                        "tooltip": "Core AI depth model. Models from the coreai-catalog.",
                    },
                ),
                "compute_unit": (
                    ["auto", "gpu", "neuralEngine", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Neural Engine runs independently from GPU — "
                        "use 'neuralEngine' to free the GPU for diffusion.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "estimate_depth"
    CATEGORY = "CoreAI/Vision"

    def estimate_depth(self, image, model: str, compute_unit: str = "auto"):
        """Run depth estimation and return the depth map as a ComfyUI tensor."""
        # 1. Save input image as PNG
        input_path = tensor_to_png(image)

        try:
            # 2. Call the runner
            runner = get_runner()
            result = runner.predict(
                model_id=model,
                image_path=input_path,
                compute_unit=compute_unit,
            )

            # 3. Load output
            output_path = result["output"].get("outputPath")
            if not output_path:
                raise RuntimeError(
                    f"Runner returned no outputPath. Full response: {result}"
                )

            depth_tensor = load_output_image(output_path)

            # Log timing
            timing = result.get("timing", {})
            ms = timing.get("totalMs", 0)
            unit = timing.get("computeUnitUsed", "?")
            logger.info(
                "CoreAI Depth [%s] %s: %.1fms (%s)",
                model,
                "OK" if depth_tensor is not None else "FAIL",
                ms,
                unit,
            )

            # Cleanup output file
            cleanup_temp(output_path)

            return (depth_tensor,)

        finally:
            cleanup_temp(input_path)
