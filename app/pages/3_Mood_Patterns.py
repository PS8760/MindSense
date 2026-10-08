"""Mood patterns: DASS severity estimate, intervention
sliders with before/after comparison, and a session-only mood tracker.

Associational-not-causal warnings are mandatory (Section 8, page 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import fusion, sidebar, theme, widgets  # noqa: E402
from mindsense import inference  # noqa: E402
from mindsense.data.dass import DEPRESSION_ITEMS, severity_band  # noqa: E402
from mindsense.screening.dass import DASS_INSTRUCTION, DASS_ITEM_TEXTS  # noqa: E402

st.set_page_config(page_title="Mood patterns — MindSense", page_icon="🌙", layout="wide")
theme.apply()
sidebar.render_sidebar()

st.title("🌙 Mood patterns")
st.caption("Gentle depression-severity check · intervention what-if sliders · session mood tracker")
st.info(
    "Everything here is **associational, not causal** — sliders show how an "
    "estimate changes with different inputs, not what will happen if you "
    "change your life.",
    icon="ℹ️",
)

if fusion.crisis_flag():
    widgets.crisis_banner(fusion.crisis_flag()["reasons"])

# --------------------------------------------------------------------------- #
# (a) severity screener
# --------------------------------------------------------------------------- #
st.subheader("a · Depression-severity screener")

info = inference.prognosis_info()
use_model = bool(info["selected_items"]) and info["model_available"]

if use_model:
    st.caption(
        f"Compact screener: {len(info['selected_items'])} items "
        "chosen by the prognosis model (ready)."
    )
    item_numbers = info["selected_items"]
else:
    st.caption(
        "The trained compact screener isn't available yet — showing the "
        "**official DASS depression subscale** (14 items, public-domain instrument) "
        "with reference scoring instead. Run ``make train`` to enable the model "
        "estimate with confidence."
    )
    item_numbers = DEPRESSION_ITEMS

st.caption(DASS_INSTRUCTION)

answers: dict[int, int] = {}
for number in item_numbers:
    key = f"ms_dass_{number}"
    value = st.select_slider(
        DASS_ITEM_TEXTS[number],
        options=[0, 1, 2, 3],
        value=0,
        key=key,
        format_func=lambda v: (
            f"{v} — {['did not apply', 'some degree', 'considerable', 'very much'][v]}"
        ),
    )
    answers[number] = int(value)

c1, c2, c3 = st.columns([1, 1, 1])
demo_age = c1.number_input("Age", min_value=5, max_value=120, value=22, step=1, key="ms_prog_age")
demo_gender = c2.selectbox("Gender", ["male", "female", "other"], key="ms_prog_gender")
run_severity = c3.button("Run screener", type="primary")

if run_severity:
    try:
        # instrument reference: sum of the answered depression items
        reference_sum = sum(answers.values())
        reference_band = severity_band("depression", reference_sum)
        st.session_state["ms_severity"] = {
            "reference_sum": reference_sum,
            "reference_band": reference_band,
            "model": inference.predict_severity(
                answers, {"age": float(demo_age), "gender": demo_gender}
            ),
            "items": list(item_numbers),
            "mode": "model" if use_model else "instrument",
        }
        model = st.session_state["ms_severity"]["model"]
        if model.get("available"):
            fusion.record(
                "prognosis",
                f"depression band {model['band']}"
                + (f" ({model['confidence']:.0%} conf.)" if model.get("confidence") else ""),
                tone="warn" if "severe" in str(model["band"]).lower() else "info",
            )
        else:
            fusion.record(
                "prognosis",
                f"reference DASS band {reference_band} ({reference_sum}/42)",
                tone="warn" if "severe" in reference_band.lower() else "info",
            )
    except ValueError as exc:
        st.error(str(exc))

severity = st.session_state.get("ms_severity")
if severity:
    col_ref, col_model = st.columns(2, gap="large")
    with col_ref:
        st.markdown("**Instrument reference scoring**")
        widgets.kpi(str(severity["reference_sum"]), "sum of answered items (0–42 scale if all 14)")
        widgets.band_chip(severity["reference_band"])
        st.caption("Official DASS bands for the raw sum (no ×2 rescaling — see DECISIONS).")
    with col_model:
        st.markdown("**Model estimate**")
        model = severity["model"]
        if not model.get("available"):
            widgets.unavailable(model)
        else:
            widgets.kpi(model["band"], "predicted severity band")
            if model.get("confidence") is not None:
                widgets.kpi(f"{model['confidence']:.0%}", "confidence (max class probability)")
            widgets.band_chip(model["band"])
            st.caption(f"model: `{model['model_version']}` · items: {model['selected_items']}")
    st.caption(
        "Screening aid only — severity bands describe questionnaire responses, "
        "not a clinical diagnosis. A qualified professional makes diagnoses."
    )
    st.divider()

# --------------------------------------------------------------------------- #
# (b) what-if sliders
# --------------------------------------------------------------------------- #
st.subheader("b · Intervention what-if")
st.caption(
    "Move the sliders to see how the estimated intervention-likelihood changes "
    "relative to a baseline profile."
)

echo: dict = {}
risk_result = st.session_state.get("ms_risk_result") or {}
if risk_result.get("input_echo"):
    echo = risk_result["input_echo"]

default_profile = {
    "stress_pressure_score": float(echo.get("stress_pressure_score") or 3.0),
    "sleep_hours": float(echo.get("sleep_hours") or 7.0),
    "work_study_hours": float(echo.get("work_study_hours") or 6.0),
    "financial_stress": float(echo.get("financial_stress") or 2.0),
    "family_history": float(echo.get("family_history") or 0.0),
    "diet_quality": float(echo.get("diet_quality") or 1.0),
}
if echo:
    st.caption("Baseline pre-filled from your Risk Assessment answers (session only).")

s1, s2, s3 = st.columns(3)
with s1:
    wi_stress = st.slider(
        "What-if: stress/pressure", 0.0, 5.0, default_profile["stress_pressure_score"], 0.5
    )
    wi_sleep = st.slider("What-if: sleep (hours)", 3.0, 12.0, default_profile["sleep_hours"], 0.5)
with s2:
    wi_hours = st.slider(
        "What-if: study/work hours/day", 0.0, 13.0, default_profile["work_study_hours"], 0.5
    )
    wi_financial = st.slider(
        "What-if: financial stress", 0.0, 5.0, default_profile["financial_stress"], 0.5
    )
with s3:
    wi_diet = st.select_slider(
        "What-if: diet quality",
        options=[0.0, 1.0, 2.0],
        value=default_profile["diet_quality"],
        format_func=lambda v: {0.0: "Unhealthy", 1.0: "Moderate", 2.0: "Healthy"}[v],
    )
    wi_population = st.radio(
        "Population", ["student", "professional"], horizontal=True, key="ms_wi_pop"
    )

what_if_profile = {
    "population": wi_population,
    "age": float(echo.get("age") or demo_age),
    "gender": str(echo.get("gender") or demo_gender),
    "stress_pressure_score": wi_stress,
    "sleep_hours": wi_sleep,
    "work_study_hours": wi_hours,
    "financial_stress": wi_financial,
    "family_history": default_profile["family_history"],
    "diet_quality": wi_diet,
    "support_available": None,
    "satisfaction_score": None,
}
baseline_profile = {
    **default_profile,
    "population": wi_population,
    "age": what_if_profile["age"],
    "gender": what_if_profile["gender"],
    "support_available": None,
    "satisfaction_score": None,
}

if st.button("Compare baseline vs what-if", type="primary"):
    try:
        before = inference.predict_intervention(baseline_profile)
        after = inference.predict_intervention(what_if_profile)
        st.session_state["ms_whatif"] = {"before": before, "after": after}
    except ValueError as exc:
        st.error(str(exc))

whatif = st.session_state.get("ms_whatif")
if whatif:
    before, after = whatif["before"], whatif["after"]
    if not before.get("available") or not after.get("available"):
        widgets.unavailable(after if not after.get("available") else before)
    else:
        fig_df = pd.DataFrame(
            {
                "scenario": ["Baseline", "What-if"],
                "intervention-likelihood": [before["likelihood"], after["likelihood"]],
            }
        )
        fig = go.Figure(
            go.Bar(
                x=fig_df["scenario"],
                y=fig_df["intervention-likelihood"],
                marker_color=["#9DBBE8", "#2FA88B"],
                text=[f"{v:.0%}" for v in fig_df["intervention-likelihood"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={"title": "Estimated likelihood", "range": [0, 1.05]},
            height=360,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        st.plotly_chart(fig, width="stretch")
        delta = after["likelihood"] - before["likelihood"]
        st.markdown(
            f"**Change:** {delta:+.0%} ({before['likelihood']:.0%} → {after['likelihood']:.0%})"
        )
        st.warning(
            "**Associational, not causal.** The model was trained on observational "
            "survey data — a higher likelihood here does not prove that changing "
            "these inputs causes a different outcome for you personally.",
            icon="⚠️",
        )
        fusion.record(
            "intervention",
            f"what-if {after['likelihood']:.0%} vs baseline {before['likelihood']:.0%}",
            tone="info",
        )

st.divider()

# --------------------------------------------------------------------------- #
# (c) session-only mood tracker
# --------------------------------------------------------------------------- #
st.subheader("c · Mood tracker (session only)")
st.caption("Optional log kept in this browser session only — close the tab and it's gone.")

mood_log: list[int] = st.session_state.setdefault("ms_mood_log", [])
mc1, mc2, mc3 = st.columns([2, 1, 1])
new_mood = mc1.select_slider(
    "How do you feel right now?",
    options=[1, 2, 3, 4, 5],
    value=3,
    format_func=lambda v: {
        1: "😞 very low",
        2: "🙁 low",
        3: "😐 okay",
        4: "🙂 good",
        5: "😄 great",
    }[v],
    key="ms_mood_value",
)
if mc2.button("Log mood", type="primary"):
    mood_log.append(int(new_mood))
if mc3.button("Clear tracker") and mood_log:
    mood_log.clear()
    st.rerun()

if mood_log:
    st.line_chart(pd.DataFrame({"mood": mood_log}), height=220)
    st.caption(f"{len(mood_log)} entry(ies) this session · values 1 (very low) – 5 (great).")
else:
    st.caption("No entries yet.")

st.divider()
st.caption(
    "Prognosis details, item-selection rationale and metrics: **Models & data** (once trained)."
)
