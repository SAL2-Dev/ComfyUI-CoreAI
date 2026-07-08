"""
nodes/chat.py — CoreAI Chat node for ComfyUI.

On-device LLM chat via the coreai-runner EngineFactory pipeline.
Supports multi-turn conversation, prefix cache (101× TTFT speedup),
and streaming token display.

Models: Qwen3 0.6B–8B, Gemma 3/4, Mistral, LFM2, gpt-oss 20B
"""

from __future__ import annotations

import logging
from typing import Any

from .. import catalog
from ..bridge import get_runner
from ..perf import format_chat_perf, with_perf

logger = logging.getLogger("ComfyUI-CoreAI")

# Cache model list
_CHAT_MODELS: list[str] | None = None
_MODELS_FETCHED_AT: float = 0
_MODELS_TTL: float = 300  # 5-minute cache


def _get_chat_models() -> list[str]:
    import time
    global _CHAT_MODELS, _MODELS_FETCHED_AT
    if _CHAT_MODELS is None or time.monotonic() - _MODELS_FETCHED_AT > _MODELS_TTL:
        _CHAT_MODELS = catalog.model_dropdown(capability="chat")
        if not _CHAT_MODELS:
            _CHAT_MODELS = catalog.model_dropdown(capability="text-generation")
        if not _CHAT_MODELS:
            _CHAT_MODELS = [
                "qwen3-0-6b",
                "qwen3-1-7b",
                "qwen3-4b",
                "gemma-3-1b",
                "gemma-3-2b",
                "lfm2-350m",
            ]
        _MODELS_FETCHED_AT = time.monotonic()
    return _CHAT_MODELS


class CoreAIChat:
    """
    On-device LLM Chat using Apple Core AI.

    Generates text from a prompt using a local language model running
    entirely on-device via the Core AI pipelined engine with prefix
    cache for fast multi-turn conversations.

    Common uses:
      - Prompt expansion / rewriting
      - Question answering
      - Code generation
      - Summarization
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _get_chat_models()
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "Explain quantum entanglement in one paragraph.",
                        "multiline": True,
                        "tooltip": "Your message to the model.",
                    },
                ),
                "model": (
                    models,
                    {
                        "default": models[0] if models else "qwen3-0-6b",
                        "tooltip": "Which LLM to use. Smaller models are faster; "
                        "larger models are smarter.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional system instruction that shapes the model's "
                        "behavior (e.g. 'You are a helpful assistant that answers concisely.')",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 256,
                        "min": 1,
                        "max": 8192,
                        "step": 16,
                        "tooltip": "Maximum number of tokens to generate.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Higher = more creative, lower = more deterministic. "
                        "0 = greedy (always pick most likely token).",
                    },
                ),
            },
            "optional": {
                "context": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Previous conversation context (for multi-turn). "
                        "Connect the 'text' output of a previous CoreAI Chat node.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run_chat"
    CATEGORY = "CoreAI/Text"

    def run_chat(
        self,
        prompt: str,
        model: str = "qwen3-0-6b",
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        context: str = "",
    ) -> tuple[str]:
        """Run the LLM and return generated text."""
        runner = get_runner()

        # Build the conversation from context + new prompt.
        # If context exists, prepend it so the model has multi-turn history.
        # The runner's prefix cache will reuse the KV from prior turns.
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\nUser: {prompt}"
        elif system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}"

        try:
            result = runner.chat(
                model_id=model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Log performance stats if available
            usage = result.get("usage", {})
            logger.info(
                "CoreAI Chat [%s]: %d chars, tokens=%s",
                model,
                len(text),
                usage.get("total_tokens", "?"),
            )

            return with_perf((text,), format_chat_perf(result))

        except Exception as e:
            error_str = str(e).lower()
            if "framework" in error_str or "absent in this sdk" in error_str:
                raise RuntimeError(
                    f"'{model}' requires a framework not available in this macOS/SDK version."
                )
            if "not_installed" in error_str or "model_load_failed" in error_str:
                raise RuntimeError(
                    f"Model '{model}' is not downloaded yet. "
                    f"Use the Download button on this node or the "
                    f"CoreAI Model Loader to install it first."
                )
            raise
