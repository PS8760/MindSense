"""Calmer chatbot + mood-lifting content tests — fully offline.

Groq is disabled suite-wide by ``conftest``; when a test needs the AI path it
monkeypatches ``groq.enabled`` and stubs the cached call so no network is used.
"""

from __future__ import annotations

import pytest

from app.components import ai
from mindsense import groq


def test_calmer_offline_fallback() -> None:
    result = ai.calmer_reply([{"role": "user", "content": "I feel anxious tonight"}])
    assert result["source"] == "local"
    assert result["reply"]
    assert result["thinking"]
    assert result["crisis"]["triggered"] is False


def test_calmer_flags_crisis() -> None:
    result = ai.calmer_reply([{"role": "user", "content": "I want to kill myself"}])
    assert result["crisis"]["triggered"] is True
    assert result["crisis"]["reasons"]


def test_calmer_uses_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq, "enabled", lambda: True)
    monkeypatch.setattr(
        ai,
        "_cached_calmer",
        lambda _h: {"reply": "You're okay. Breathe.", "thinking": "Validate, then ground."},
    )
    result = ai.calmer_reply([{"role": "user", "content": "hi"}])
    assert result["source"] == "groq"
    assert result["reply"] == "You're okay. Breathe."
    assert result["thinking"] == "Validate, then ground."


def test_calmer_failure_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq, "enabled", lambda: True)

    def _boom(_h):
        raise RuntimeError("network down")

    monkeypatch.setattr(ai, "_cached_calmer", _boom)
    result = ai.calmer_reply([{"role": "user", "content": "hi"}])
    assert result["source"] == "local"
    assert result["reply"]


def test_uplift_offline_has_items() -> None:
    result = ai.uplift_content(2)
    assert result["source"] == "local"
    assert len(result["items"]) >= 4
    for item in result["items"]:
        assert item["title"]
        assert item["body"]


def test_uplift_uses_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq, "enabled", lambda: True)
    monkeypatch.setattr(
        ai,
        "_cached_uplift",
        lambda _m: {"items": [{"kind": "Breathing", "title": "Breathe", "body": "Slow."}]},
    )
    result = ai.uplift_content(4)
    assert result["source"] == "groq"
    assert result["items"][0]["title"] == "Breathe"
