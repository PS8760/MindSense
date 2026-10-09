"""Gentle, non-clinical AI reflections powered by Groq (optional layer).

Flow: rule-based *gaps* are computed locally from the form answers (always
available, no network), then Groq turns those into warm, practical
suggestions. If the API/key/network is unavailable the component falls back
to hand-written suggestions — the page never breaks.

Secrets: ``GROQ_API_KEY`` from the environment or ``.env`` (git-ignored).
Only the numeric answers the user already sees are sent — no names, no free
text unless the user explicitly opts in on the text page.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")

_SYSTEM = (
    "You are MindSense, a warm, plain-language wellness companion inside a "
    "screening app. You are NOT a doctor and never diagnose. Based only on the "
    "numbers provided, write a short, kind note with exactly these parts:\n"
    "1) **What stands out** — one or two gentle observations, no blame.\n"
    "2) **Try this week** — at most 4 small, concrete, realistic actions "
    "(sleep, breaks, study/work rhythm, talking to someone, campus/HR support).\n"
    "3) **When to reach out** — one sentence: if feelings persist/worsen or "
    "safety is a concern, talk to a professional; mention crisis helplines "
    "exist in the app.\n"
    "Tone: calm, human, encouraging; short sentences; no jargon; no lists of "
    "disclaimers; max 130 words total; English."
)

_FALLBACK = """**What stands out**
Thanks for checking in — answering honestly already takes courage, and
nothing here is a diagnosis.

**Try this week**
- Protect a small, regular sleep window; even 30 extra minutes helps.
- Break study/work into short chunks with real breaks between them.
- Tell one person how you've actually been feeling lately.
- Move your body a little most days — a short walk counts.

**When to reach out**
If low mood or anxiety persists beyond two weeks, gets worse, or you ever
feel unsafe, please talk to a professional — the **Help now** button has
verified helplines."""


def groq_key() -> str | None:
    """Resolve the Groq API key from the environment or the repo ``.env``."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        from dotenv import load_dotenv

        from mindsense.utils.io import repo_path

        env_path = repo_path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return os.environ.get("GROQ_API_KEY", "").strip() or None
    except Exception:  # noqa: BLE001 - key discovery must never crash the UI
        return None
    return None


def _chat(system: str, user: str, *, temperature: float = 0.6, max_tokens: int = 1024) -> str:
    """One Groq chat-completions call; tries both models before raising.

    ``gpt-oss`` models are *reasoning* models: they spend tokens on the
    ``reasoning`` field before producing ``content``. A small ``max_tokens``
    budget is exhausted by reasoning, leaving ``content`` empty and the call
    looking like a failure. We therefore give enough budget for reasoning +
    the reply, and fall back to the ``reasoning`` text when ``content`` is
    still empty.
    """
    key = groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY not configured")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload_base = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error = "groq request failed"
    for model in _MODELS:
        try:
            response = requests.post(
                GROQ_URL,
                headers=headers,
                json={**payload_base, "model": model},
                timeout=25,
            )
        except requests.RequestException as exc:  # network down → offline path
            raise RuntimeError(f"network error: {exc}") from exc
        if response.status_code == 200:
            message = response.json()["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if not content:  # reasoning model used its budget before the answer
                content = (message.get("reasoning") or "").strip()
            if content:
                return content
        last_error = f"HTTP {response.status_code}: {response.text[:180]}"
        if response.status_code in (400, 404):  # unknown model → try the other
            continue
        break
    raise RuntimeError(last_error)


# --------------------------------------------------------------------------- #
# rule-based gaps (always available)
# --------------------------------------------------------------------------- #
def compute_gaps(features: dict[str, Any], quiz: dict[str, Any] | None) -> list[dict[str, str]]:
    """Local, deterministic 'what stands out' cards from the raw answers.

    Returns dicts with ``icon``, ``title`` and ``detail`` — gentle wording,
    never diagnostic.
    """

    def num(key: str) -> float | None:
        value = features.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    gaps: list[dict[str, str]] = []

    def add(icon: str, title: str, detail: str) -> None:
        gaps.append({"icon": icon, "title": title, "detail": detail})

    sleep = num("sleep_hours")
    if sleep is not None and sleep < 6.5:
        add(
            "🌙",
            "Sleep is running low",
            f"You're averaging about {sleep:g}h — small earlier nights can steady mood and focus.",
        )
    elif sleep is not None and sleep > 9.5:
        add(
            "😴",
            "Extra-long sleep",
            f"About {sleep:g}h nightly can sometimes signal low mood too — worth noticing if it lasts.",
        )

    stress = num("stress_pressure_score")
    if stress is not None and stress >= 4:
        add(
            "🌊",
            "Pressure is high",
            "Your stress rating is near the top of the scale — that takes a real toll over time.",
        )

    if features.get("population") == "student":
        hours = num("work_study_hours")
        if hours is not None and hours >= 10:
            add(
                "📚",
                "Very long days",
                f"About {hours:g}h of study/work daily leaves little room to recover.",
            )
        satisfaction = num("satisfaction_score")
        if satisfaction is not None and satisfaction <= 1:
            add(
                "🧭",
                "Low satisfaction",
                "You're rating satisfaction near zero — a good signal to reassess load or expectations.",
            )
        financial = num("financial_stress")
        if financial is not None and financial >= 4:
            add(
                "💼",
                "Money pressure",
                "Financial stress is a common amplifier of sleep and mood strain.",
            )
        diet = num("diet_quality")
        if diet is not None and diet == 0:
            add(
                "🥗",
                "Eating habits",
                "Rough eating routines can quietly worsen energy and mood — small swaps count.",
            )
    else:
        support = num("support_available")
        if support is not None and support == 0:
            add(
                "🏢",
                "Little workplace support",
                "You reported no mental-health support at work — check if an EAP or HR channel exists.",
            )
        elif support is not None and support == 0.5:
            add(
                "❓",
                "Support is unclear",
                "You weren't sure what support exists — a quick look at HR pages could clarify it.",
            )

    if float(features.get("family_history") or 0) >= 1:
        add(
            "🌳",
            "Family history",
            "Family history raises awareness, not destiny — extra attention to early signs is wise.",
        )

    if quiz:
        phq, gad = quiz["phq9"], quiz["gad7"]
        if phq["total"] >= 15:
            add(
                "💙",
                "Low mood is prominent",
                f"PHQ-9 {phq['total']}/27 ({phq['band']}) — this level deserves support, not just willpower.",
            )
        elif phq["total"] >= 10:
            add(
                "🌤️",
                "Low mood is showing up",
                f"PHQ-9 {phq['total']}/27 ({phq['band']}) — early enough that small changes help a lot.",
            )
        elif phq["total"] >= 5:
            add(
                "🌱",
                "Mild mood dip",
                f"PHQ-9 {phq['total']}/27 ({phq['band']}) — worth tracking for a week or two.",
            )
        if gad["total"] >= 15:
            add(
                "🫁",
                "Anxiety is loud",
                f"GAD-7 {gad['total']}/21 ({gad['band']}) — constant worry is exhausting; techniques + support help.",
            )
        elif gad["total"] >= 10:
            add(
                "🌬️",
                "Anxiety is noticeable",
                f"GAD-7 {gad['total']}/21 ({gad['band']}) — catching it now keeps it manageable.",
            )
        elif gad["total"] >= 5:
            add(
                "🍃",
                "Some nervousness",
                f"GAD-7 {gad['total']}/21 ({gad['band']}) — normal range, but keep an eye on it.",
            )

    if not gaps:
        add(
            "✨",
            "Nothing stands out",
            "Your answers look steady today — a good moment to build habits before stress piles up.",
        )
    return gaps


# --------------------------------------------------------------------------- #
# Groq suggestions
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_suggestions(context_json: str) -> str:
    context = json.loads(context_json)
    return _chat(_SYSTEM, json.dumps(context, ensure_ascii=False))


def suggestions(context: dict[str, Any]) -> dict[str, str]:
    """Personalised note for the check-in results.

    Returns ``{"text": ..., "source": "groq" | "local"}`` — never raises.
    """
    try:
        text = _cached_suggestions(json.dumps(context, sort_keys=True))
        return {"text": text, "source": "groq"}
    except Exception:  # noqa: BLE001 - any failure → calm offline fallback
        return {"text": _FALLBACK, "source": "local"}


def render_suggestions(context: dict[str, Any]) -> dict[str, str]:
    """Spinner + suggestion card. Returns the result dict (for tests/pages)."""
    with st.spinner("Thinking about your answers…"):
        result = suggestions(context)
    st.markdown(result["text"])
    if result["source"] == "groq":
        st.caption(
            "✨ Written just now by an AI assistant (Groq · Llama) from only the "
            "numbers above — nothing stored, no personal identity sent."
        )
    else:
        st.caption(
            "📴 Offline suggestions (AI unavailable right now) — the same "
            "practical next steps, written by us."
        )
    return result


_SYSTEM_TEXT = (
    "You are MindSense, a warm wellness companion. A person just shared how "
    "they've been feeling in their own words. Write a gentle, brief reflection "
    "(max 80 words): one validating observation, two small suggestions, and one "
    "sentence pointing to professional help if feelings persist or safety is a "
    "concern. Never diagnose. Plain language, kind tone, English."
)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_reflection(text: str) -> str:
    return _chat(_SYSTEM_TEXT, text, temperature=0.7, max_tokens=320)


def reflection(text: str) -> dict[str, str]:
    """AI reflection on free text — opt-in only (the UI asks first)."""
    try:
        return {"text": _cached_reflection(text), "source": "groq"}
    except Exception:  # noqa: BLE001
        return {
            "text": (
                "Thanks for sharing that. I can't reach the AI assistant right "
                "now — but what you wrote sounds like it deserves a real "
                "conversation with someone you trust. If it persists or "
                "worsens, please reach out to a professional (see **Help now**)."
            ),
            "source": "local",
        }
