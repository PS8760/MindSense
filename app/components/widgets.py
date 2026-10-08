"""Reusable result widgets: gauges, chips, banners, bars, empty states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go
import streamlit as st

TIER_TAGS = {
    "lower": ("✓", "Lower concern", "ms-tag-lower"),
    "elevated": ("▲", "Elevated", "ms-tag-elevated"),
    "higher": ("■", "Higher", "ms-tag-higher"),
}

#: Severity words share one visual language across PHQ-9 / GAD-7 / DASS.
_SEVERE_WORDS = ("severe", "extremely", "moderately severe", "high", "crisis")
_MILD_WORDS = ("mild", "moderate", "elevated", "medium")


def tier_chip(tier: str) -> None:
    """Tier shown as symbol + word + colour (never colour alone)."""
    symbol, word, css = TIER_TAGS.get(tier, ("•", tier.title(), "ms-tag-info"))
    st.markdown(
        f'<span class="ms-tag {css}">{symbol} {word}</span>',
        unsafe_allow_html=True,
    )


def band_chip(band: str) -> None:
    """Severity band chip with an icon derived from the wording."""
    lowered = band.lower()
    if any(word in lowered for word in _SEVERE_WORDS):
        css, icon = "ms-tag-higher", "■"
    elif any(word in lowered for word in _MILD_WORDS):
        css, icon = "ms-tag-elevated", "▲"
    else:
        css, icon = "ms-tag-lower", "✓"
    st.markdown(f'<span class="ms-tag {css}">{icon} {band}</span>', unsafe_allow_html=True)


def kpi(value: str, label: str) -> None:
    st.markdown(
        f'<div class="ms-kpi">{value}</div><div class="ms-kpi-label">{label}</div>',
        unsafe_allow_html=True,
    )


def crisis_banner(reasons: Sequence[str]) -> None:
    """Persistent, non-dismissible crisis banner with immediate next step."""
    reason_text = "; ".join(reasons) if reasons else "a crisis indicator"
    st.error(
        f"**Support is recommended right now — {reason_text}.**\n\n"
        "You are not alone. Please reach out to a crisis line (use **Help now** in "
        "the sidebar), contact a trusted person, or contact local emergency services. "
        "MindSense is a screening aid and cannot provide help itself.",
        icon="🚨",
    )


def unavailable(result: Mapping[str, Any]) -> None:
    """Render the graceful empty state for ``available=False`` results."""
    st.info(result.get("message", "This module is not available yet."))


def probability_bars(probabilities: Mapping[str, float], *, title: str = "") -> None:
    """Horizontal probability bars, sorted descending."""
    items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    labels = [name for name, _ in items]
    values = [value for _, value in items]
    colours = ["#3B7DD8" if i == 0 else "#9DBBE8" for i in range(len(items))]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title or "Predicted signal probabilities",
        template="plotly_white",
        xaxis={"title": "Probability", "range": [0, 1.05]},
        yaxis={"autorange": "reversed"},
        height=max(260, 60 * len(items) + 140),
        margin={"l": 10, "r": 10, "t": 50 if title else 30, "b": 10},
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def risk_gauge(probability: float, tier: str) -> None:
    """0–1 risk gauge with a plain-language tier label underneath."""
    symbol, word, _ = TIER_TAGS.get(tier, ("•", tier.title(), "ms-tag-info"))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "", "valueformat": ".0%", "font": {"size": 40, "color": "#1F2A37"}},
            title={
                "text": f"{symbol} {word} concern",
                "font": {"size": 20, "color": "#1F2A37"},
            },
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickformat": ".0%"},
                "bar": {"color": "#3B7DD8", "thickness": 0.35},
                "steps": [
                    {"range": [0, 0.35], "color": "#E3F5EF"},
                    {"range": [0.35, 0.65], "color": "#FDF3E3"},
                    {"range": [0.65, 1], "color": "#FBE9E8"},
                ],
                "threshold": {
                    "line": {"color": "#C0504D", "width": 4},
                    "thickness": 0.9,
                    "value": probability,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_white", height=320, margin={"l": 30, "r": 30, "t": 70, "b": 10}
    )
    st.plotly_chart(fig, width="stretch")


def contributions_chart(contributions: Sequence[Mapping[str, Any]]) -> None:
    """Approximate feature contributions as a diverging bar chart."""
    if not contributions:
        return
    items = sorted(contributions, key=lambda item: abs(float(item["contribution"])))
    labels = [str(item["feature"]).replace("_", " ") for item in items]
    values = [float(item["contribution"]) for item in items]
    colours = ["#2FA88B" if v >= 0 else "#D95F5F" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colours))
    fig.update_layout(
        title="What pushed the estimate (approximate contribution)",
        template="plotly_white",
        xaxis={"title": "Contribution to score"},
        height=max(280, 42 * len(items) + 130),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def entity_chips(entities: Sequence[Mapping[str, Any]]) -> None:
    """Render extracted entities as coloured chips with their labels."""
    if not entities:
        st.caption("No clinical entities found in this text.")
        return
    html = "".join(
        f'<span class="ms-tag ms-tag-info" title="{e["label"]}">{e["text"]} · {e["label"]}</span>'
        for e in entities
    )
    st.markdown(html, unsafe_allow_html=True)


def next_steps(tier: str) -> None:
    """Plain-language, non-diagnostic next steps for a risk tier."""
    st.markdown("**Next steps**")
    common = [
        "Share how you've been feeling with someone you trust.",
        "If these feelings persist or worsen, consider booking an "
        "appointment with a doctor or counsellor.",
    ]
    if tier == "higher":
        st.markdown(
            "\n".join(
                f"- {item}"
                for item in [
                    "Your answers suggest a higher level of concern — please "
                    "consider professional support soon; you don't need to wait "
                    "until things feel unmanageable.",
                    *common,
                ]
            )
        )
    elif tier == "elevated":
        st.markdown(
            "\n".join(
                f"- {item}"
                for item in [
                    "Some of your answers point to elevated stress or low mood — "
                    "a check-in with a professional can help clarify what's going on.",
                    *common,
                ]
            )
        )
    else:
        st.markdown(
            "\n".join(
                f"- {item}"
                for item in [
                    "Your answers look reassuring right now — keep the habits "
                    "that support you (sleep, movement, connection).",
                    *common,
                ]
            )
        )
    st.caption(
        "These are general wellness suggestions from public health guidance, not medical advice."
    )
