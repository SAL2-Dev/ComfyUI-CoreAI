"""
Unit tests for the model-picker dropdown logic (offline — no network).

Locks the behaviour that replaced the broken `readiness_score` sort: the dropdown
hard-gates to models that run on this Mac and orders them smallest-first, so the
lightest usable model is the default (never an unobtainable or F-grade entry).
"""

from __future__ import annotations

from comfyui_coreai import catalog


def test_artifact_size_bytes_parsing():
    f = catalog._artifact_size_bytes
    assert f({"size": {"artifact_size": "36.2MB"}}) == 36.2 * 1e6
    assert f({"size": {"artifact_size": "1.7 GB"}}) == 1.7 * 1e9
    assert f({"size": {"artifact_size": "969MB"}}) == 969 * 1e6
    # unknown / unparseable / not_published sort last
    assert f({"size": {"artifact_size": "not_published"}}) == float("inf")
    assert f({"size": {}}) == float("inf")
    assert f({}) == float("inf")


def test_runs_on_this_mac_gate():
    g = catalog._runs_on_this_mac
    base = {"artifact": {"availability": "available"}, "size": {"artifact_size": "50MB"}}
    # available + mac -> keep
    assert g({**base, "device_support": {"mac": True}})
    # unknown mac -> keep (do NOT penalize under-curated/community entries)
    assert g({**base, "device_support": {"mac": "unknown"}})
    # missing device_support -> keep
    assert g(base)
    # explicitly not mac -> drop
    assert not g({**base, "device_support": {"mac": False}})
    # not_published artifact -> drop (cannot download)
    assert not g({"device_support": {"mac": True}, "size": {"artifact_size": "not_published"}})
    # unavailable artifact -> drop
    assert not g({"artifact": {"availability": "planned"},
                  "device_support": {"mac": True}, "size": {"artifact_size": "50MB"}})


def test_model_dropdown_orders_smallest_first_and_gates(monkeypatch):
    fake = [
        {"id": "big", "name": "Big", "size": {"artifact_size": "4GB"},
         "device_support": {"mac": True}, "artifact": {"availability": "available"}},
        {"id": "small", "name": "Small", "size": {"artifact_size": "50MB"},
         "device_support": {"mac": True}, "artifact": {"availability": "available"}},
        {"id": "mid", "name": "Mid", "size": {"artifact_size": "800MB"},
         "device_support": {"mac": "unknown"}, "artifact": {"availability": "available"}},
        {"id": "iphone-only", "name": "iOnly", "size": {"artifact_size": "20MB"},
         "device_support": {"mac": False}, "artifact": {"availability": "available"}},
        {"id": "unpublished", "name": "Soon", "size": {"artifact_size": "not_published"},
         "device_support": {"mac": True}, "artifact": {"availability": "available"}},
    ]
    monkeypatch.setattr(catalog, "list_models", lambda capability=None: list(fake))
    ids = catalog.model_dropdown(capability="whatever")
    # smallest-first; iphone-only and unpublished are gated out
    assert ids == ["small", "mid", "big"]
