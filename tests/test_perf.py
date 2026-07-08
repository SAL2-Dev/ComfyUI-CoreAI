"""
Tests for the perf stats display (Tier 2.4).

Covers:
  - perf.py formatters (pure, no runner): vision + chat, cached badge, edge cases
  - node-level ui output: each inference node returns ``{"ui": {"coreai_perf": ...}}``
    alongside its ``result`` tuple, driven by the mock runner's timing data.
  - mock runner chat response now carries timing + streaming_stats so the chat
    node's tok/s badge has real data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_RUNNER = REPO_ROOT / "tools" / "mock_runner.py"


# --- Pure formatter tests (no runner, no torch) -----------------------------

def test_format_vision_perf_inference_ms_and_unit():
    from comfyui_coreai.perf import format_vision_perf

    timing = {
        "load_ms": 1.0, "preprocess_ms": 0.0, "inference_ms": 5.0,
        "postprocess_ms": 0.5, "total_ms": 6.5, "compute_unit_used": "GPU",
    }
    perf = format_vision_perf(timing)
    assert perf["text"] == "5.0ms · GPU"
    assert perf["inference_ms"] == 5.0
    assert perf["compute_unit"] == "GPU"
    assert perf["label_ms"] == "inference"


def test_format_vision_perf_falls_back_to_total_ms():
    from comfyui_coreai.perf import format_vision_perf

    # No inference_ms — should use total_ms and label it "total".
    timing = {"total_ms": 17.4, "compute_unit_used": "neuralEngine"}
    perf = format_vision_perf(timing)
    assert perf["text"] == "17.4ms · neuralEngine"
    assert perf["inference_ms"] == 17.4
    assert perf["label_ms"] == "total"


def test_format_vision_perf_missing_unit():
    from comfyui_coreai.perf import format_vision_perf

    perf = format_vision_perf({"inference_ms": 8.0})
    assert perf["compute_unit"] == "?"
    assert perf["text"] == "8.0ms · ?"


def test_format_chat_perf_computes_tps_and_no_cached():
    from comfyui_coreai.perf import format_chat_perf

    result = {
        "timing": {"inference_ms": 40.0, "total_ms": 42.5, "compute_unit_used": "GPU"},
        "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
        "streaming_stats": {"tokens_per_second": 200.0, "reused_prompt_tokens": 0},
    }
    perf = format_chat_perf(result)
    # Prefers the runner's measured tokens_per_second.
    assert perf["tokens_per_second"] == 200.0
    assert perf["reused_prompt_tokens"] == 0
    assert perf["text"] == "200 tok/s · 40.0ms · GPU"
    assert perf["completion_tokens"] == 8


def test_format_chat_perf_derives_tps_when_streaming_absent():
    from comfyui_coreai.perf import format_chat_perf

    # No streaming_stats and no tokens_per_second — derive from completion_tokens / inference_ms.
    result = {
        "timing": {"inference_ms": 50.0, "compute_unit_used": "GPU"},
        "usage": {"completion_tokens": 10, "prompt_tokens": 4},
    }
    perf = format_chat_perf(result)
    # 10 tokens / 0.05s = 200 tok/s
    assert perf["tokens_per_second"] == 200.0
    assert perf["reused_prompt_tokens"] == 0


def test_format_chat_perf_cached_badge_when_reused():
    from comfyui_coreai.perf import format_chat_perf

    result = {
        "timing": {"inference_ms": 40.0, "compute_unit_used": "GPU"},
        "usage": {"completion_tokens": 8, "prompt_tokens": 120},
        "streaming_stats": {"tokens_per_second": 200.0, "reused_prompt_tokens": 95},
    }
    perf = format_chat_perf(result)
    assert perf["reused_prompt_tokens"] == 95
    # Text still contains the tok/s headline.
    assert "200 tok/s" in perf["text"]


def test_with_perf_wraps_result_and_ui():
    from comfyui_coreai.perf import format_vision_perf, with_perf

    perf = format_vision_perf({"inference_ms": 5.0, "compute_unit_used": "GPU"})
    out = with_perf(("depth",), perf)
    assert out["result"] == ("depth",)
    assert out["ui"]["coreai_perf"] is perf


# --- Mock runner chat response now carries timing ---------------------------

def test_chat_response_has_timing_and_streaming_stats(runner):
    result = runner.chat(
        model_id="qwen3-0-6b",
        messages=[{"role": "user", "content": "hello world"}],
        max_tokens=16,
    )
    assert "timing" in result
    assert "inference_ms" in result["timing"]
    assert "compute_unit_used" in result["timing"]
    assert result["timing"]["compute_unit_used"] == "GPU"
    assert "streaming_stats" in result
    assert "tokens_per_second" in result["streaming_stats"]
    assert "reused_prompt_tokens" in result["streaming_stats"]


# --- Node-level ui output (against the mock runner) -------------------------

def _tiny_image():
    """A numpy [C,H,W] float32 image in [0,1] — works without torch."""
    return np.random.rand(3, 16, 16).astype("float32")


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


def _perf_of(ret):
    """Extract the coreai_perf payload from a node's {result, ui} return."""
    assert isinstance(ret, dict), f"node returned {type(ret)}; expected dict"
    assert "result" in ret and "ui" in ret
    return ret["ui"]["coreai_perf"]


def test_depth_node_returns_perf(runner):
    from comfyui_coreai.nodes.depth import CoreAIDepthEstimation

    node = CoreAIDepthEstimation()
    ret = node.estimate_depth(_tiny_image(), model="depth-anything-3-small")
    perf = _perf_of(ret)
    assert "ms" in perf["text"]
    assert perf["compute_unit"] in ("GPU", "gpu", "neuralEngine", "auto")
    assert "inference_ms" in perf


def test_detection_node_returns_perf(runner):
    from comfyui_coreai.nodes.detection import CoreAIObjectDetection

    node = CoreAIObjectDetection()
    ret = node.detect_objects(_tiny_image(), model="rf-detr-nano")
    perf = _perf_of(ret)
    assert "ms" in perf["text"]


def test_instance_seg_node_returns_perf(runner):
    from comfyui_coreai.nodes.instance_seg import CoreAIInstanceSegmentation

    node = CoreAIInstanceSegmentation()
    ret = node.segment_instances(_tiny_image(), model="rf-detr-seg-nano")
    perf = _perf_of(ret)
    assert "ms" in perf["text"]


def test_vlm_node_returns_perf_with_tps(runner):
    from comfyui_coreai.nodes.vlm import CoreAIVisionLanguage

    node = CoreAIVisionLanguage()
    ret = node.run_vlm(
        _tiny_image(), model="qwen3-vl-2b", prompt="Describe this.", max_tokens=32,
    )
    perf = _perf_of(ret)
    assert perf["tokens_per_second"] is not None
    assert "tok/s" in perf["text"]


def test_chat_node_returns_perf_with_tps(runner):
    from comfyui_coreai.nodes.chat import CoreAIChat

    node = CoreAIChat()
    ret = node.run_chat(prompt="What is the capital of France?", model="qwen3-0-6b")
    perf = _perf_of(ret)
    assert perf["tokens_per_second"] is not None
    assert "tok/s" in perf["text"]
    assert "reused_prompt_tokens" in perf


def test_embedding_node_returns_perf(runner):
    from comfyui_coreai.nodes.embedding import CoreAIImageTextSimilarity

    node = CoreAIImageTextSimilarity()
    ret = node.compute_similarity(
        _tiny_image(),
        captions="a cat\na dog\na car",
        model="official-clip-vit-base-patch32",
    )
    perf = _perf_of(ret)
    assert "ms" in perf["text"]


def test_image_gen_node_returns_perf(runner):
    from comfyui_coreai.nodes.image_gen import CoreAIImageGeneration

    node = CoreAIImageGeneration()
    ret = node.generate(prompt="a serene landscape", model="official-flux-2-klein-4b")
    perf = _perf_of(ret)
    assert "ms" in perf["text"]


def test_segmentation_node_returns_perf(runner):
    from comfyui_coreai.nodes.segmentation import CoreAISegmentation

    node = CoreAISegmentation()
    ret = node.segment(
        _tiny_image(), model="official-sam-3", text_prompt="cat", score_threshold=0.3,
    )
    perf = _perf_of(ret)
    assert "ms" in perf["text"]
