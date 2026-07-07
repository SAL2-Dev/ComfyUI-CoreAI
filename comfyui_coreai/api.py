"""
api.py — ComfyUI API route registration.

Proxies /coreai/* routes to the runner's Unix socket so the JavaScript
web extension can talk to the runner through ComfyUI's HTTP server.
The user never needs to know about the Unix socket.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ComfyUI-CoreAI")


def register_routes() -> None:
    """Register /coreai/* API routes on the ComfyUI PromptServer."""

    from server import PromptServer
    from aiohttp import web
    from .bridge import get_runner
    from . import catalog as catalog_client

    @PromptServer.instance.routes.get("/coreai/health")
    async def coreai_health(request):
        try:
            runner = get_runner()
            health = runner.health()
            return web.json_response(health)
        except Exception as e:
            return web.json_response(
                {"error": str(e), "status": "unavailable"},
                status=200,
            )

    @PromptServer.instance.routes.get("/coreai/models")
    async def coreai_models(request):
        capability = request.query.get("capability")
        try:
            runner = get_runner()
            models = runner.list_models(capability=capability)
            return web.json_response({"models": models})
        except Exception:
            # Fall back to catalog-only (runner not started yet)
            models = catalog_client.list_models(capability=capability)
            return web.json_response({
                "models": models,
                "source": "catalog",
                "runner": "offline",
            })

    @PromptServer.instance.routes.get("/coreai/models/{model_id}/status")
    async def coreai_model_status(request):
        model_id = request.match_info["model_id"]
        try:
            runner = get_runner()
            status = runner.model_status(model_id)
            return web.json_response(status)
        except Exception as e:
            return web.json_response({
                "model_id": model_id,
                "installed": False,
                "loaded": False,
                "error": str(e),
            })

    @PromptServer.instance.routes.post("/coreai/models/{model_id}/download")
    async def coreai_model_download(request):
        model_id = request.match_info["model_id"]
        try:
            runner = get_runner()
            result = runner.download_model(model_id)
            return web.json_response(result)
        except Exception as e:
            return web.json_response(
                {"error": str(e), "model_id": model_id},
                status=500,
            )

    @PromptServer.instance.routes.get("/coreai/catalog/model/{model_id}")
    async def coreai_catalog_model(request):
        model_id = request.match_info["model_id"]
        info = catalog_client.get_model(model_id)
        if not info:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(info)

    logger.info("ComfyUI-CoreAI API routes registered: /coreai/*")
