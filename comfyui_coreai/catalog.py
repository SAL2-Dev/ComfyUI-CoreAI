"""
catalog.py — Client for the coreai-catalog API.
Caches responses for 5 minutes so dropdown population is instant after
the first load. The catalog is the source of truth for what models exist
and what they can do — node UIs read from here.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger("ComfyUI-CoreAI")

CATALOG_API = "https://raw.githubusercontent.com/kevinqz/coreai-catalog/main/dist"
CACHE_TTL = 300  # 5 minutes

# Module-level cache (in-process, shared across all nodes)
_cache: dict[str, tuple[Any, float]] = {}


def _get_cached(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and time.monotonic() - entry[1] < CACHE_TTL:
        return entry[0]
    return None


def _set_cached(key: str, value: Any) -> None:
    _cache[key] = (value, time.monotonic())


def list_models(
    capability: str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    List models from the catalog, optionally filtered by capability and/or device.
    Returns a list of model entry dicts.

    The catalog is distributed as static JSON on GitHub Pages:
      https://raw.githubusercontent.com/kevinqz/coreai-catalog/main/dist/catalog.json

    On network failure, returns the stale cache (or empty list if no cache).
    """
    cache_key = f"models:{capability or ''}:{device or ''}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    # Fetch the full catalog JSON (it's a static file, not a queryable API)
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{CATALOG_API}/catalog.json")
            resp.raise_for_status()
            data = resp.json()
            # catalog.json has { "models": [...] }
            models = data.get("models", data) if isinstance(data, dict) else data
    except Exception as e:
        logger.warning("Catalog fetch failed: %s — returning cached data", e)
        return cached or []

    # Apply filters client-side
    filtered = models
    if capability:
        filtered = [m for m in filtered if capability in (m.get("capabilities") or [])]
    if device:
        def ds(m):
            return m.get("device_support") or {}
        if device == "mac":
            filtered = [m for m in filtered if ds(m).get("mac") or ds(m).get("mac_only")]
        elif device == "iphone":
            filtered = [m for m in filtered if ds(m).get("iphone")]
        elif device == "ipad":
            filtered = [m for m in filtered if ds(m).get("ipad")]

    _set_cached(cache_key, filtered)
    return filtered


def get_model(model_id: str) -> dict[str, Any] | None:
    """Get a single model's metadata from the catalog."""
    cache_key = f"model:{model_id}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    # The catalog is a flat JSON file — fetch and filter
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{CATALOG_API}/catalog.json")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", data) if isinstance(data, dict) else data
            for m in models:
                if m.get("id") == model_id:
                    _set_cached(cache_key, m)
                    return m
    except Exception as e:
        logger.warning("Catalog fetch failed for '%s': %s", model_id, e)

    return cached


_SIZE_UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def _artifact_size_bytes(model: dict[str, Any]) -> float:
    """Parse ``size.artifact_size`` ('969MB', '54.5 MB', '4.01GB') into bytes so
    the dropdown can sort smallest-first. Unknown/unparseable sizes sort last."""
    raw = ((model.get("size") or {}).get("artifact_size") or "").strip()
    match = re.match(r"([\d.]+)\s*([KMGT]?B)\b", raw, re.IGNORECASE)
    if not match:
        return float("inf")
    return float(match.group(1)) * _SIZE_UNITS.get(match.group(2).upper(), 1)


def _runs_on_this_mac(model: dict[str, Any]) -> bool:
    """Hard gate for the picker: keep models actually usable on Apple Silicon
    macOS. Drops only entries whose artifact is not available or that are
    explicitly marked non-mac. ``device_support.mac == 'unknown'`` (and missing)
    is KEPT on purpose — under-curated/community entries must not be penalized."""
    if (model.get("artifact") or {}).get("availability") not in (None, "available"):
        return False
    if (model.get("size") or {}).get("artifact_size") == "not_published":
        return False  # explicit "can't download this yet" marker
    if (model.get("device_support") or {}).get("mac") is False:
        return False
    return True


def model_dropdown(capability: str | None = None) -> list[str]:
    """
    Get a list of model IDs for populating ComfyUI dropdowns.

    Filtered by capability, hard-gated to models that run on this Mac (available
    artifact, not explicitly non-mac), then sorted smallest-first so the lightest
    usable model is the default — an honest, capability-blind order.

    We deliberately do NOT sort by the catalog's ``readiness_score``: it is a
    curation/deployability composite that is blind to model quality (it inversely
    tracks capability), and it is not emitted per-entry on catalog.json anyway,
    so the old sort silently collapsed to alphabetical and surfaced the LEAST
    ready model as the default. See the SotA red-team notes.
    """
    models = [m for m in list_models(capability=capability) if _runs_on_this_mac(m)]
    models.sort(key=lambda m: (_artifact_size_bytes(m), m.get("name", "")))
    return [m["id"] for m in models if m.get("id")]


def model_info_badge(model_id: str) -> str:
    """
    Build a human-readable info string for a model:
    "54.5 MB · fp16 · 15ms · ANE"

    Used in ComfyUI node UI to show model metadata inline.
    """
    model = get_model(model_id)
    if not model:
        return model_id

    parts: list[str] = []

    size = model.get("size", {})
    if artifact := size.get("artifact_size"):
        parts.append(artifact)

    if precision := size.get("precision"):
        parts.append(precision)

    runtime = model.get("runtime", {})
    if runner := runtime.get("runner"):
        runner_short = runner.replace("CoreAI", "").strip()
        if runner_short:
            parts.append(runner_short)

    # License badge
    license_info = model.get("license", {})
    if license_name := license_info.get("name"):
        if license_name == "Apache-2.0":
            pass  # don't clutter — Apache is the default
        elif "check_license" == license_info.get("commercial_use"):
            parts.append("check license")

    return f"{model_id}  ({' · '.join(parts)})" if parts else model_id


def invalidate_cache() -> None:
    """Clear all cached catalog data."""
    _cache.clear()
