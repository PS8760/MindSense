"""Unit tests for the rule-based gap analysis and the Groq AI fallback layer."""

from __future__ import annotations

import pytest

from app.components import ai


def _student_features() -> dict:
    return {
        "population": "student",
        "age": 22.0,
        "gender": "male",
        "sleep_hours": 5.5,
        "stress_pressure_score": 4.0,
        "satisfaction_score": 4.0,
        "work_study_hours": 8.0,
        "financial_stress": 1.0,
        "family_history": 0.0,
        "diet_quality": 1.0,
    }


def test_gaps_surface_low_sleep_and_stress() -> None:
    gaps = ai.compute_gaps(_student_features(), None)
    titles = [g["title"] for g in gaps]
    assert any("Sleep" in t for t in titles)
    assert any("Pressure" in t for t in titles)


def test_gaps_workplace_support_none() -> None:
    features = {
        "population": "professional",
        "age": 34.0,
        "gender": "female",
        "support_available": 0.0,
        "stress_pressure_score": 2.0,
        "sleep_hours": 7.0,
        "family_history": 0.0,
    }
    gaps = ai.compute_gaps(features, None)
    titles = [g["title"] for g in gaps]
    assert any("workplace support" in t.lower() for t in titles)


def test_gaps_pick_up_severe_phq() -> None:
    quiz = {
        "phq9": {"total": 18, "band": "Moderately severe"},
        "gad7": {"total": 4, "band": "Minimal"},
    }
    gaps = ai.compute_gaps(_student_features(), quiz)
    titles = [g["title"] for g in gaps]
    assert any("Low mood is prominent" in t for t in titles)


def test_gaps_quiet_answers_get_reassurance() -> None:
    features = {
        "population": "student",
        "sleep_hours": 7.5,
        "stress_pressure_score": 1.0,
        "work_study_hours": 4.0,
        "satisfaction_score": 4.0,
        "financial_stress": 1.0,
        "family_history": 0.0,
        "diet_quality": 1.0,
    }
    gaps = ai.compute_gaps(features, None)
    assert any("Nothing stands out" in g["title"] for g in gaps)


def test_gaps_handle_string_inputs() -> None:
    features = {"population": "student", "sleep_hours": "5.0", "stress_pressure_score": "3"}
    gaps = ai.compute_gaps(features, None)
    assert any("low" in g["title"].lower() for g in gaps)


def test_chat_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai, "groq_key", lambda: None)
    with pytest.raises(RuntimeError):
        ai._chat("system", "user")


def test_suggestions_fall_back_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(ai, "_cached_suggestions", _boom)
    result = ai.suggestions({"gaps": ["Sleep is running low"]})
    assert result["source"] == "local"
    assert "reach out" in result["text"].lower()


def test_groq_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    assert ai.groq_key() == "gsk-test-key"
