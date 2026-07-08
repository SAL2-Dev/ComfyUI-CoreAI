"""
perf.py — Performance stats formatting for CoreAI nodes.

The runner returns timing/usage data on every response. This module
formats it into a compact, human-readable string for display in the
ComfyUI node UI (the perf badge widget added by web/extensions/coreai_browser.js).

Two flavours:
  - vision:   "5.0ms · GPU"
  - chat/VLM: "191 tok/s · 5.0ms · GPU"  (+ cached badge when prefix reuse fires)

The badge also exposes a structured `ui` dict so the web extension can
render a green "N cached" badge when ``reused_prompt_tokens > 0``.

Kept torch-free and side-effect-free so it is trivially unit-testable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ComfyUI-CoreAI")


def format_vision_perf(timing: dict[str, Any]) -> dict[str, Any]:
    """Build the perf UI payload for a vision/image node.

    Shows ``inference_ms`` + ``compute_unit_used`` (e.g. "5.0ms · GPU").
    Falls back to ``total_ms`` when ``inference_ms`` is absent.
    """
    ms = timing.get("inference_ms")
    label_ms = "inference"
    if ms is None:
        ms = timing.get("total_ms", 0)
        label_ms = "total"
    unit = timing.get("compute_unit_used") or "?"

    text = f"{_fmt_ms(ms)} · {unit}"
    return {
        "text": text,
        "inference_ms": ms,
        "label_ms": label_ms,
        "compute_unit": unit,
    }


def format_chat_perf(result: dict[str, Any]) -> dict[str, Any]:
    """Build the perf UI payload for a chat / VLM node.

    Computes ``tok/s`` from ``usage.completion_tokens`` and
    ``timing.inference_ms``, and flags prefix-cache reuse via
    ``reused_prompt_tokens`` when > 0.

    For chat responses (OpenAI shape) timing may live under ``timing``;
    streaming stats (``tokens_per_second``, ``reused_prompt_tokens``)
    may live under ``streaming_stats`` or at the top level.
    """
    timing = result.get("timing", {}) or {}
    usage = result.get("usage", {}) or {}
    streaming = result.get("streaming_stats") or {}

    completion_tokens = usage.get("completion_tokens", 0) or 0
    prompt_tokens = usage.get("prompt_tokens", 0) or 0

    # Prefer the runner's measured tokens_per_second (streaming) when present.
    tps = streaming.get("tokens_per_second") or result.get("tokens_per_second")
    ms = timing.get("inference_ms")
    if ms is None:
        ms = timing.get("total_ms", 0)

    if tps is None and ms and ms > 0:
        tps = completion_tokens / (ms / 1000.0)
    if tps is not None:
        tps = round(float(tps), 1)

    # Prefix-cache reuse badge (green when > 0)
    reused = (
        streaming.get("reused_prompt_tokens")
        or result.get("reused_prompt_tokens")
        or 0
    )
    reused = int(reused or 0)

    unit = timing.get("compute_unit_used") or "?"

    parts: list[str] = []
    if tps is not None:
        parts.append(f"{tps:g} tok/s")
    if ms:
        parts.append(f"{_fmt_ms(ms)}")
    if parts:
        parts.append(unit)
    text = " · ".join(parts) if parts else ""

    return {
        "text": text,
        "tokens_per_second": tps,
        "completion_tokens": int(completion_tokens),
        "prompt_tokens": int(prompt_tokens),
        "reused_prompt_tokens": reused,
        "inference_ms": ms,
        "compute_unit": unit,
    }


def with_perf(
    result_tuple: tuple,
    perf: dict[str, Any],
) -> dict[str, Any]:
    """Wrap a node's return tuple into the ComfyUI ``{"ui": ..., "result": ...}``
    shape so the perf data is sent to the frontend as a hidden UI output without
    adding a visible output slot.

    ``result_tuple`` is the original tuple the node would have returned (mapping
    to ``RETURN_TYPES``). ``perf`` is the dict produced by
    :func:`format_vision_perf` or :func:`format_chat_perf`.
    """
    return {"result": result_tuple, "ui": {"coreai_perf": perf}}


def _fmt_ms(ms: float | int | None) -> str:
    """Format a millisecond value with one decimal place."""
    if ms is None:
        return "—"
    try:
        return f"{float(ms):.1f}ms"
    except (TypeError, ValueError):
        return "—"
