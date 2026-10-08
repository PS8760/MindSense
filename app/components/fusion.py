"""Session-scoped module fusion (Section 8, fusion summary).

Modules are independent: each records one plain-language line. When more
than one module has been used the sidebar/Home render the lines together
with an explicit note — MindSense never produces a single fused score.
Crisis triggers from any module raise a persistent session banner.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

_STATE_KEY = "ms_fusion"
_CRISIS_KEY = "ms_crisis"

#: Deterministic presentation order (stable across pages).
MODULE_ORDER = [
    "questionnaire",
    "risk",
    "text",
    "prognosis",
    "intervention",
    "face",
]


def _state() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault(_STATE_KEY, {})


def record(module: str, summary: str, *, tone: str = "info") -> None:
    """Record one module's plain-language outcome for the fusion summary."""
    if tone not in {"info", "warn", "crisis"}:
        raise ValueError(f"tone must be info|warn|crisis, got {tone!r}")
    _state()[module] = {"summary": summary, "tone": tone}


def modules_used() -> list[str]:
    state = _state()
    ordered = [name for name in MODULE_ORDER if name in state]
    ordered += [name for name in state if name not in MODULE_ORDER]
    return ordered


def crisis_flag() -> dict[str, Any] | None:
    return st.session_state.get(_CRISIS_KEY)


def raise_crisis(source: str, reason: str) -> None:
    """Mark the session as crisis-flagged (persistent, non-dismissible)."""
    current = st.session_state.get(_CRISIS_KEY, {"reasons": [], "sources": []})
    if reason not in current["reasons"]:
        current["reasons"].append(reason)
    if source not in current["sources"]:
        current["sources"].append(source)
    st.session_state[_CRISIS_KEY] = current


def reset() -> None:
    st.session_state.pop(_STATE_KEY, None)
    st.session_state.pop(_CRISIS_KEY, None)


def summary_lines() -> list[dict[str, Any]]:
    """Fusion lines in deterministic module order."""
    state = _state()
    return [{"module": name, **state[name]} for name in modules_used()]


def render(*, compact: bool = False) -> None:
    """Render the fusion summary (compact = sidebar; full = Home)."""
    lines = summary_lines()
    if not lines:
        return
    label = {
        "questionnaire": "Questionnaire",
        "risk": "Lifestyle risk",
        "text": "Text check-in",
        "prognosis": "Prognosis",
        "intervention": "What-if",
        "face": "Face cue",
    }
    icon = {"info": "•", "warn": "▲", "crisis": "●"}
    if len(lines) == 1:
        st.markdown(f"**This session:** {lines[0]['summary']}")
        return
    st.markdown("**Session summary (modules are independent)**")
    for line in lines:
        name = label.get(line["module"], line["module"].title())
        mark = icon.get(line["tone"], "•")
        st.markdown(f"{mark} **{name}:** {line['summary']}")
    if not compact:
        st.caption(
            "Each module runs independently on different inputs; MindSense does not "
            "combine them into one score. Treat this as a conversation starter with a "
            "professional, not a diagnosis."
        )
