#!/usr/bin/env python3
"""
mock_runner.py — a DEV/TEST stand-in for the real ``coreai-runner`` Swift binary.

The real runner (kevinqz/coreai-runner) cannot be compiled on this machine yet:
CoreAIKit requires macOS 27.0+ (the `CoreAI` framework), which ships only with
macOS 27 (beta). This mock lets the whole ComfyUI-CoreAI Python path — bridge,
nodes, image round-trip — run end-to-end TODAY with placeholder outputs.

It speaks the exact same wire protocol as the Swift server:
  * Unix domain socket, HTTP/1.1, JSON bodies.
  * launched as ``mock_runner.py --socket <path>``.
  * writes ``<socket>.ready`` containing its PID once the socket is listening.
  * routes: GET /v1/health, GET /v1/models, POST /v1/predict,
            POST /v1/models/{id}/load, POST /v1/models/{id}/unload.
  * request/response JSON uses snake_case keys — the SotA convention for this
    API (matches coreai-catalog, ComfyUI, and the HF/LLM ecosystem). The real
    Swift runner maps these to its camelCase Swift properties via explicit
    snake_case CodingKeys. ``kind`` VALUES (e.g. "depthMap") stay verbatim —
    they are the runner's enum values, not wire keys.

Zero third-party dependencies (stdlib only, incl. a tiny pure-Python PNG writer),
so it runs under any python3 — including Linux CI, where AF_UNIX works fine.

To use it instead of the real binary, point the bridge at it:
    export COREAI_RUNNER_PATH="$PWD/tools/mock_runner.py"
To switch back to the real binary later, ``unset COREAI_RUNNER_PATH`` (or point
it at the compiled ``coreai-runner``).
"""

from __future__ import annotations

import json
import os
import platform
import signal
import socketserver
import struct
import sys
import tempfile
import threading
import uuid
import zlib
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

DEFAULT_SOCKET = "/tmp/coreai-runner.sock"


# --- Tiny pure-Python PNG writer (grayscale, 8-bit) ------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_gray_png(path: str, width: int, height: int, *, fill: int = 128,
                   gradient: bool = False) -> None:
    """Write a minimal valid 8-bit grayscale PNG with stdlib only."""
    width = max(1, int(width))
    height = max(1, int(height))
    raw = bytearray()
    for _y in range(height):
        raw.append(0)  # filter type 0 (None) per scanline
        if gradient:
            row = bytes(int(255 * x / max(1, width - 1)) for x in range(width))
        else:
            row = bytes([fill & 0xFF]) * width
        raw.extend(row)
    idat = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)


def png_size(path: str | None) -> tuple[int, int]:
    """Read width/height from a PNG's IHDR; fall back to 512x512."""
    if path:
        try:
            with open(path, "rb") as handle:
                head = handle.read(24)
            if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
                width, height = struct.unpack(">II", head[16:24])
                return int(width), int(height)
        except OSError:
            pass
    return 512, 512


def _tmp_png(prefix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex}.png")


# --- Capability routing (mirrors the node fallback model ids) --------------

def capability_for(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "rf-detr-seg" in mid:
        return "instance-segmentation"
    if "depth" in mid:
        return "monocular-depth"
    if "clip" in mid:
        return "image-text-similarity"
    if "sam" in mid:  # official-sam-3, efficientsam3-tinyvit
        return "promptable-segmentation"
    if "flux" in mid or "z-image" in mid:
        return "image-generation"
    if "qwen" in mid or "minicpm" in mid or "vl" in mid:
        return "vision-language"
    if "rf-detr" in mid or "yolox" in mid:
        return "object-detection"
    return "object-detection"


def _mock_detections() -> list[dict]:
    return [
        {"label": "object", "score": 0.95, "bbox": [0.10, 0.10, 0.50, 0.60]},
        {"label": "object", "score": 0.82, "bbox": [0.55, 0.20, 0.90, 0.80]},
    ]


def _compute_unit_label(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    return "GPU"


# --- Response builders (camelCase — matches Codables.swift) -----------------

def build_predict_response(body: dict) -> dict:
    model_id = body.get("model_id", "")
    inp = body.get("input") or {}
    options = body.get("options") or {}
    capability = capability_for(model_id)
    image_path = inp.get("image_path")
    width, height = png_size(image_path)

    if capability == "monocular-depth":
        out_path = _tmp_png("coreai_mock_depth")
        write_gray_png(out_path, width, height, gradient=True)
        output = {"kind": "depthMap", "output_path": out_path}
    elif capability == "image-generation":
        out_path = _tmp_png("coreai_mock_gen")
        write_gray_png(out_path, 512, 512, gradient=True)
        output = {"kind": "image", "output_path": out_path}
    elif capability in ("object-detection", "instance-segmentation"):
        output = {"kind": "detections", "detections": _mock_detections()}
    elif capability == "vision-language":
        prompt = (inp.get("prompt") or "").strip()
        output = {"kind": "text",
                  "text": f"[mock] A placeholder description for prompt: {prompt[:80]!r}"}
    elif capability == "promptable-segmentation":
        masks = []
        for i in range(2):
            mask_path = _tmp_png("coreai_mock_mask")
            write_gray_png(mask_path, width, height, fill=255)
            masks.append({
                "mask_path": mask_path,
                "score": round(0.95 - 0.10 * i, 3),
                "bbox": [0.10 + 0.10 * i, 0.10, 0.50 + 0.10 * i, 0.60],
            })
        output = {"kind": "masks", "mask_paths": masks}
    elif capability == "image-text-similarity":
        captions = (inp.get("prompt") or "").split("|||")
        scores = [round(max(0.0, 0.90 - 0.15 * i), 4) for i in range(len(captions))]
        output = {"kind": "text", "text": json.dumps(scores)}
    else:
        output = {"kind": "detections", "detections": _mock_detections()}

    timing = {
        "load_ms": 1.0,
        "preprocess_ms": 0.0,
        "inference_ms": 5.0,
        "postprocess_ms": 0.5,
        "total_ms": 6.5,
        "compute_unit_used": _compute_unit_label(options.get("compute_unit")),
    }
    return {"model_id": model_id, "output": output, "timing": timing}


def build_health_response() -> dict:
    return {
        "status": "ok",
        "device": platform.node() or "Mac",
        "chip": "Apple Silicon (mock runner)",
        "memory_total_gb": 24.0,
        "memory_available_gb": 12.0,
        "macos_version": platform.mac_ver()[0] or "26.0",
        "coreai_version": "mock-0.1",
        "loaded_models": [],
        "thermal_state": "nominal",
    }


def _model_entry(mid: str, name: str, family: str, cap: str) -> dict:
    return {
        "id": mid, "name": name, "family": family, "capability": cap,
        "device_support": ["mac"], "installed": False, "loaded": False,
    }


def build_models_response() -> dict:
    return {"models": [
        _model_entry("depth-anything-3-small", "Depth Anything 3 Small",
                     "depth-anything-3", "monocular-depth"),
        _model_entry("rf-detr-nano", "RF-DETR Nano", "rf-detr", "object-detection"),
        _model_entry("qwen3-vl-2b", "Qwen3-VL 2B", "qwen3-vl", "vision-language"),
    ]}


# --- HTTP handler -----------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence default stderr access logging
        pass

    def _send_json(self, obj: dict, code: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        path = urlparse(self.path).path
        if path == "/v1/health":
            self._send_json(build_health_response())
        elif path == "/v1/models":
            self._send_json(build_models_response())
        else:
            self._send_json({"error": {"code": "NOT_FOUND", "message": path}}, 404)

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        path = urlparse(self.path).path
        try:
            if path == "/v1/predict":
                self._send_json(build_predict_response(self._read_json()))
            elif path.startswith("/v1/models/") and path.endswith("/load"):
                model_id = path[len("/v1/models/"):-len("/load")]
                self._send_json({"model_id": model_id, "status": "loaded"})
            elif path.startswith("/v1/models/") and path.endswith("/unload"):
                model_id = path[len("/v1/models/"):-len("/unload")]
                self._send_json({"model_id": model_id, "status": "unloaded"})
            else:
                self._send_json({"error": {"code": "NOT_FOUND", "message": path}}, 404)
        except Exception as exc:  # noqa: BLE001 (mock: report any error as 500)
            self._send_json(
                {"error": {"code": "INTERNAL", "message": str(exc)}}, 500)


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def server_bind(self):
        # Remove a stale socket left by a previous (crashed) run before binding.
        try:
            os.unlink(self.server_address)
        except OSError:
            pass
        super().server_bind()


# --- Entry point ------------------------------------------------------------

def parse_socket_path(argv: list[str]) -> str:
    i = 0
    path = DEFAULT_SOCKET
    while i < len(argv):
        arg = argv[i]
        if arg in ("--socket", "-s") and i + 1 < len(argv):
            path = argv[i + 1]
            i += 2
            continue
        i += 1
    return path


def main(argv: list[str]) -> int:
    socket_path = parse_socket_path(argv)
    ready_path = socket_path + ".ready"

    server = _UnixHTTPServer(socket_path, _Handler)  # binds + listens here

    stop = threading.Event()

    def _handle_signal(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Signal readiness the same way the real runner does: PID in <socket>.ready.
    with open(ready_path, "w") as handle:
        handle.write(str(os.getpid()))

    try:
        stop.wait()
    finally:
        server.shutdown()
        server.server_close()
        for path in (socket_path, ready_path):
            try:
                os.unlink(path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
