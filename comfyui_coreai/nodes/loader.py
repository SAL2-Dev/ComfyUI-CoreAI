"""
nodes/loader.py — CoreAI Model Loader / Downloader utility node.

Provides a UI for browsing, downloading, and managing Core AI models
from the catalog. Not an inference node — it's a utility that other
nodes reference.

Usage in workflow: connect to a CoreAI inference node's model input
to pre-download and pin a specific model.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import catalog
from ..bridge import get_runner

logger = logging.getLogger("ComfyUI-CoreAI")


class CoreAIModelLoader:
    """
    Model Loader / Downloader for Core AI models.

    Lists all models from the catalog, grouped by capability.
    Pre-downloads the selected model so the first inference is instant.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        all_models = catalog.model_dropdown()
        if not all_models:
            all_models = ["depth-anything-3-small"]

        return {
            "required": {
                "model": (
                    all_models,
                    {
                        "default": all_models[0],
                        "tooltip": "Model ID from the coreai-catalog.",
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download the model on first execution if not cached.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_id",)
    FUNCTION = "load_model"
    CATEGORY = "CoreAI/Utils"

    def load_model(self, model: str, auto_download: bool = True):
        """Return the model ID, optionally pre-downloading."""
        if auto_download:
            try:
                runner = get_runner()
                info = runner.load_model(model)
                logger.info("Model '%s' loaded: %s", model, info.get("status"))
            except Exception as e:
                logger.warning("Could not pre-load model '%s': %s", model, e)

        return (model,)


class CoreAIHealthCheck:
    """
    Health Check for the Core AI runner.

    Returns device info, loaded models, thermal state, and memory.
    Use this to diagnose issues or verify the runner is working.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "check_health"
    CATEGORY = "CoreAI/Utils"

    def check_health(self):
        """Check runner health and return formatted status."""
        try:
            runner = get_runner()
            health = runner.health()

            import json
            status = (
                f"Device: {health.get('device', '?')}\n"
                f"Chip: {health.get('chip', '?')}\n"
                f"Memory: {health.get('memory_available_gb', 0):.1f} / "
                f"{health.get('memory_total_gb', 0):.1f} GB free\n"
                f"OS: {health.get('macos_version', '?')}\n"
                f"Core AI: {health.get('coreai_version', '?')}\n"
                f"Thermal: {health.get('thermal_state', '?')}\n"
                f"Loaded models: {', '.join(health.get('loaded_models', [])) or 'none'}"
            )
            return (status,)
        except Exception as e:
            return (f"Runner error: {e}",)
