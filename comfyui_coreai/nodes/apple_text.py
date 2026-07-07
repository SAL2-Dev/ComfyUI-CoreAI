"""
nodes/apple_text.py — CoreAI Apple Text (FoundationModels).

REAL on-device text generation using Apple's SYSTEM language model
(FoundationModels, macOS 26+, Apple Intelligence enabled), via the
tools/fm-generate Swift CLI.

This is distinct from the coreai-catalog `.aimodel` vision models (depth, SAM,
detection, VLM, image-gen) — those run on the coreai-runner over Core AI and need
macOS 27's `CoreAI` framework. FoundationModels ships with macOS 26 and is
text-only, so this node runs TODAY without the runner or macOS 27. Useful for
generating / expanding prompts for the diffusion node, on-device and private.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("ComfyUI-CoreAI")


def _fm_binary() -> str | None:
    """Locate the compiled fm-generate CLI (env override, then repo tools/)."""
    env = os.environ.get("COREAI_FM_PATH")
    if env and Path(env).exists():
        return env
    # comfyui_coreai/nodes/apple_text.py -> repo root -> tools/fm-generate
    cand = Path(__file__).resolve().parents[2] / "tools" / "fm-generate"
    if cand.exists() and os.access(cand, os.X_OK):
        return str(cand)
    return None


class CoreAIAppleText:
    """On-device text generation with Apple's FoundationModels (macOS 26+).

    Text-only (FoundationModels has no image input on macOS 26). Requires Apple
    Intelligence enabled; the system model downloads on first use.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "Write a vivid one-line image prompt: a serene mountain lake at golden hour.",
                        "multiline": True,
                        "tooltip": "Apple's on-device text model (FoundationModels). "
                        "Runs on macOS 26+ with Apple Intelligence on — no coreai-runner needed.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "CoreAI/On-Device"

    def generate(self, prompt: str):
        binary = _fm_binary()
        if not binary:
            return (
                "[FoundationModels backend not built — run tools/build_fm.sh "
                "(macOS 26+, Apple Silicon).]",
            )
        try:
            result = subprocess.run(
                [binary, "--prompt", prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("fm-generate invocation failed: %s", e)
            return (f"[fm-generate failed: {e}]",)

        if result.returncode == 0:
            return (result.stdout.strip(),)
        if result.returncode == 3:
            return (
                f"[Apple model unavailable — {result.stderr.strip()}. Enable Apple "
                "Intelligence in System Settings; the model downloads on first use.]",
            )
        return (f"[Apple model error: {result.stderr.strip() or result.returncode}]",)
