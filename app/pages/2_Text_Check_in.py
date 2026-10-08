"""Text Check-in (Exp 7 + 5): free-text signal, entities, crisis detection.

Text stays in ``st.session_state`` only — never logged, never persisted.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

from app.components import fusion, sidebar, theme, widgets  # noqa: E402
from mindsense import inference  # noqa: E402

st.set_page_config(page_title="Text Check-in — MindSense", page_icon="💬", layout="wide")
theme.apply()
sidebar.render_sidebar()

st.title("💬 Text Check-in")
st.caption("Mental-state signal from your words (Exp 7) · clinical entities (Exp 5)")
st.info(
    "Whatever you type is analyzed **only in this browser session** — it is "
    "never stored, logged or sent anywhere.",
    icon="🔒",
)

if fusion.crisis_flag():
    widgets.crisis_banner(fusion.crisis_flag()["reasons"])

mode = st.radio(
    "Mode",
    ["Check-in signal", "Clinical-style note (entity extraction)"],
    horizontal=True,
    help="Check-in = class probabilities from the text model. "
    "Clinical-style note = structured entity extraction (medications, "
    "conditions, symptoms) as a NLP demonstration.",
)

placeholder = (
    "Example: “Lately I can't sleep, I've lost interest in everything I used "
    "to enjoy, and I feel anxious about exams…”"
    if mode == "Check-in signal"
    else "Example: “Patient prescribed sertraline 50 mg for major depressive "
    "disorder; reports fatigue and occasional panic attacks, no suicidal ideation.”"
)

text = st.text_area(
    "How have you been feeling? (or paste a note)",
    height=180,
    placeholder=placeholder,
    key="ms_text_input",
)

col_run, col_clear = st.columns([1, 5])
run = col_run.button("🔎 Analyze text", type="primary")
if col_clear.button("Clear text"):
    st.session_state.pop("ms_text_input", None)
    st.session_state.pop("ms_text_result", None)
    st.session_state.pop("ms_entity_result", None)
    st.rerun()

if run:
    try:
        if mode == "Check-in signal":
            st.session_state["ms_text_result"] = inference.analyze_text(text)
        else:
            st.session_state["ms_entity_result"] = inference.extract_entities(text)
    except ValueError as exc:
        st.error(str(exc))

# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
result = st.session_state.get("ms_text_result")
if result is not None:
    st.subheader("Text signal")
    if not result.get("available"):
        widgets.unavailable(result)
        st.caption(
            "Crisis language detection still runs on your text (rules below) "
            "even without the trained model."
        )
    else:
        col_bars, col_words = st.columns([3, 2], gap="large")
        with col_bars:
            widgets.probability_bars(result["probabilities"])
        with col_words:
            st.markdown(f"**Strongest signal:** `{result['top_class']}`")
            if result["top_words"]:
                st.markdown("**Words driving it** (tf-idf × coefficient):")
                for item in result["top_words"]:
                    weight = item["weight"]
                    arrow = "↑" if weight >= 0 else "↓"
                    st.markdown(f"- {arrow} **{item['word']}** ({weight:+.2f})")
            else:
                st.caption("Individual word contributions unavailable for this model.")
            st.caption(f"model: `{result['model_version']}`")

    crisis = result.get("crisis")
    if crisis:
        if crisis["triggered"]:
            widgets.crisis_banner(crisis["reasons"])
            fusion.raise_crisis("Text", crisis["reasons"][0])
            tone = "crisis"
        else:
            tone = "warn" if crisis.get("max_suicidal_probability") else "info"
        top = (result.get("probabilities") or {}).get(
            result.get("top_class", ""), None
        )
        summary = (
            f"signal `{result.get('top_class', 'n/a')}`"
            + (f" ({top:.0%})" if top is not None else "")
            + (" · crisis language detected" if crisis["triggered"] else "")
        )
        fusion.record("text", summary, tone=tone)
        st.caption(
            f"Crisis rule: suicidal-class probability vs threshold "
            f"{crisis['threshold']:.2f} + lexicon check "
            f"({'hit' if crisis['lexicon_hits'] else 'no hits'})."
        )

entity_result = st.session_state.get("ms_entity_result")
if entity_result is not None:
    st.subheader("Extracted entities")
    if not entity_result.get("available"):
        widgets.unavailable(entity_result)
    else:
        widgets.entity_chips(entity_result["entities"])
        st.caption(
            "Entity extraction (Exp 5) finds mentions of conditions, "
            "medications and symptoms — it describes *what is written*, "
            "and is never a diagnosis of the writer."
        )
        fusion.record(
            "text",
            f"{len(entity_result['entities'])} entities extracted",
            tone="info",
        )

st.divider()
st.caption(
    "Model: TF-IDF + linear classifier over labelled mental-health text "
    "(Exp 7, metrics in Lab Results). Crisis rules: probability threshold + "
    "high-precision lexicon (config/screening.crisis)."
)
