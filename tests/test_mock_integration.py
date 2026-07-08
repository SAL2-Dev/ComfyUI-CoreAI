"""
Integration test: comfyui_coreai.bridge ↔ tools/mock_runner.py over the real
Unix-socket wire protocol.

This exercises the whole Python path without the Swift binary and without torch
or a running ComfyUI: the bridge spawns the mock runner (pointed at via
COREAI_RUNNER_PATH), talks HTTP over the Unix socket, and we assert the response
shape matches the runner's camelCase contract for every capability. It runs on
Linux CI too (AF_UNIX + stdlib only).
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_RUNNER = REPO_ROOT / "tools" / "mock_runner.py"


def _write_tiny_png(path: Path, width: int = 16, height: int = 16) -> str:
    """Write a minimal valid grayscale PNG to use as node input."""
    raw = bytearray()
    for _ in range(height):
        raw.append(0)
        raw.extend([128] * width)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))
    path.write_bytes(png)
    return str(path)


def _is_png(path: str) -> bool:
    with open(path, "rb") as handle:
        return handle.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def runner():
    """A bridge runner backed by the mock, freshly spawned and torn down."""
    os.chmod(MOCK_RUNNER, 0o755)
    os.environ["COREAI_RUNNER_PATH"] = str(MOCK_RUNNER)

    from comfyui_coreai import bridge

    bridge.CoreAIRunner._instance = None  # reset singleton for a clean process
    instance = bridge.get_runner()
    try:
        yield instance
    finally:
        instance.shutdown()
        bridge.CoreAIRunner._instance = None
        os.environ.pop("COREAI_RUNNER_PATH", None)


def test_health_returns_camelcase_keys(runner):
    health = runner.health()
    assert health["status"] == "ok"
    for key in ("memory_total_gb", "memory_available_gb", "macos_version",
                "coreai_version", "thermal_state", "loaded_models"):
        assert key in health, f"missing {key}"


def test_list_models(runner):
    models = runner.list_models()
    assert isinstance(models, list) and models
    assert "id" in models[0]


def test_depth(runner, tmp_path):
    img = _write_tiny_png(tmp_path / "in.png")
    result = runner.predict(model_id="depth-anything-3-small", image_path=img)
    assert result["output"]["kind"] == "depthMap"
    assert _is_png(result["output"]["output_path"])
    assert "total_ms" in result["timing"]
    assert result["timing"]["compute_unit_used"]


def test_object_detection(runner, tmp_path):
    img = _write_tiny_png(tmp_path / "in.png")
    result = runner.predict(model_id="rf-detr-nano", image_path=img,
                            score_threshold=0.5)
    dets = result["output"]["detections"]
    assert dets and {"label", "score", "bbox"} <= set(dets[0])
    assert len(dets[0]["bbox"]) == 4


def test_vision_language(runner, tmp_path):
    img = _write_tiny_png(tmp_path / "in.png")
    result = runner.predict(model_id="qwen3-vl-2b", image_path=img,
                            prompt="Describe this.", max_tokens=32)
    assert result["output"]["kind"] == "text"
    assert result["output"]["text"]


def test_promptable_segmentation(runner, tmp_path):
    img = _write_tiny_png(tmp_path / "in.png")
    # text_prompt exercises the param that previously TypeError'd in bridge.predict
    result = runner.predict(model_id="official-sam-3", image_path=img,
                            text_prompt="cat", score_threshold=0.3)
    masks = result["output"]["mask_paths"]
    assert masks and _is_png(masks[0]["mask_path"])
    assert "score" in masks[0]


def test_instance_segmentation(runner, tmp_path):
    img = _write_tiny_png(tmp_path / "in.png")
    result = runner.predict(model_id="rf-detr-seg-nano", image_path=img,
                            score_threshold=0.5)
    assert result["output"]["detections"]


def test_clip_similarity(runner, tmp_path):
    img = _write_tiny_png(tmp_path / "in.png")
    result = runner.predict(model_id="official-clip-vit-base-patch32",
                            image_path=img, prompt="a cat|||a dog|||a car")
    scores = json.loads(result["output"]["text"])
    assert isinstance(scores, list) and len(scores) == 3


def test_image_generation(runner):
    result = runner.predict(model_id="official-flux-2-klein-4b",
                            prompt="a serene landscape")
    assert result["output"]["kind"] == "image"
    assert _is_png(result["output"]["output_path"])


def test_chat_completion(runner):
    """Test POST /v1/chat/completions — OpenAI-compatible chat."""
    result = runner.chat(
        model_id="qwen3-0-6b",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=128,
        temperature=0.7,
    )
    assert result["object"] == "chat.completion"
    assert result["model"] == "qwen3-0-6b"
    assert len(result["choices"]) == 1
    choice = result["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]
    assert choice["finish_reason"] == "stop"
    usage = result["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
