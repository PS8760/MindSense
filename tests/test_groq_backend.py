"""Groq-backend inference tests — fully offline (``chat_json`` is stubbed)."""

from __future__ import annotations

import pytest

from mindsense import groq, inference


def _enable(monkeypatch: pytest.MonkeyPatch, reply: dict) -> None:
    monkeypatch.setattr(groq, "enabled", lambda: True)
    monkeypatch.setattr(groq, "chat_json", lambda *_a, **_k: reply)


def _valid_risk() -> dict:
    return {
        "population": "student",
        "age": 22.0,
        "gender": "male",
        "sleep_hours": 5.0,
        "stress_pressure_score": 4.0,
        "work_study_hours": 9.0,
    }


def test_predict_risk_uses_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(
        monkeypatch,
        {
            "probability": 0.72,
            "contributions": [
                {"feature": "sleep_hours", "value": 5.0, "contribution": 0.3},
                {"feature": "stress_pressure_score", "value": 4.0, "contribution": 0.2},
            ],
        },
    )
    result = inference.predict_risk(_valid_risk())
    assert result["available"] is True
    assert result["source"] == "groq"
    assert result["probability"] == pytest.approx(0.72)
    assert result["tier"] == "higher"
    assert result["contributions"][0]["feature"] == "sleep_hours"
    assert result["model_version"].startswith("groq:")


def test_predict_intervention_uses_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, {"likelihood": 1.7})  # clamped to 1.0
    result = inference.predict_intervention(_valid_risk())
    assert result["available"] is True
    assert result["source"] == "groq"
    assert result["likelihood"] == 1.0
    assert result["warning"]


def test_predict_severity_uses_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch, {"band": "moderate", "confidence": 0.55})
    result = inference.predict_severity({1: 1, 2: 2, 3: 0}, {"age": 22, "gender": "male"})
    assert result["available"] is True
    assert result["source"] == "groq"
    assert result["band"] == "Moderate"  # case-normalised to a known DASS band
    assert result["confidence"] == pytest.approx(0.55)


def test_analyze_text_uses_groq_and_keeps_crisis(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(
        monkeypatch,
        {
            "top_class": "Depression",
            "probabilities": {"Depression": 0.6, "Normal": 0.4, "NotALabel": 0.9},
            "top_words": [{"word": "tired", "weight": 0.8}, {"word": "down", "weight": 0.5}],
        },
    )
    result = inference.analyze_text("I am so tired and down lately")
    assert result["available"] is True
    assert result["source"] == "groq"
    assert result["top_class"] == "Depression"
    assert "crisis" in result
    assert set(result["probabilities"]) == set(inference._TEXT_CLASSES)
    assert result["top_words"][0]["word"] == "tired"


def test_groq_availability_reports_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq, "enabled", lambda: True)
    status = inference.availability()
    assert status["risk"]["available"] is True
    assert "Groq" in status["risk"]["detail"]
    assert "Groq" not in status["face"]["detail"]  # face stays ONNX-only


def test_groq_failure_falls_back_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq, "enabled", lambda: True)

    def _boom(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(groq, "chat_json", _boom)
    risk = inference.predict_risk(_valid_risk())
    assert risk["available"] is False
    assert risk["message"]
    text = inference.analyze_text("hello world")
    assert text["available"] is False
    assert "crisis" in text


def test_chat_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq, "groq_key", lambda: None)
    with pytest.raises(RuntimeError):
        groq.chat("system", "user")


def test_chat_json_parses_fenced_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groq, "chat", lambda *_a, **_k: '```json\n{"ok": 1}\n```')
    assert groq.chat_json("system", "user") == {"ok": 1}
