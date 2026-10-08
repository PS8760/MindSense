"""Risk Assessment (Exp 6 + 8): lifestyle form + PHQ-9/GAD-7 questionnaires.

All model calls go through :mod:`mindsense.inference` (never loading models
directly). PHQ-9 item 9 > 0 raises the crisis banner immediately.
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
from mindsense.screening import gad7 as gad7_module  # noqa: E402
from mindsense.screening import phq9 as phq9_module  # noqa: E402

st.set_page_config(page_title="Risk Assessment — MindSense", page_icon="📝", layout="wide")
theme.apply()
sidebar.render_sidebar()

st.title("📝 Risk Assessment")
st.caption("Lifestyle inputs (Exp 6) with standard PHQ-9 & GAD-7 questionnaires · explainable (Exp 8)")
st.info(
    "This page estimates **concern level** from your inputs — it is a screening "
    "aid, not a diagnosis. Your answers never leave this session.",
    icon="🔒",
)

if fusion.crisis_flag():
    widgets.crisis_banner(fusion.crisis_flag()["reasons"])

# --------------------------------------------------------------------------- #
# 1. lifestyle form
# --------------------------------------------------------------------------- #
st.subheader("1 · Lifestyle inputs")

col_pop, col_demo = st.columns([1, 2])
with col_pop:
    population = st.radio(
        "I am answering as a…",
        ["student", "professional"],
        horizontal=True,
        help="Two source populations were harmonised; each field is only asked "
        "where the source survey actually collected it.",
    )
with col_demo:
    age = st.number_input("Age", min_value=5, max_value=120, value=22, step=1)
    gender = st.selectbox("Gender", ["male", "female", "other"], index=0)

st.markdown(
    f"""
Fields below match exactly what the **{population}** source survey collected —
fields the survey never asked about stay empty rather than being guessed.
"""
)

features: dict[str, object] = {
    "population": population,
    "age": float(age),
    "gender": gender,
    "sleep_hours": None,
    "stress_pressure_score": None,
    "satisfaction_score": None,
    "work_study_hours": None,
    "financial_stress": None,
    "family_history": None,
    "support_available": None,
    "diet_quality": None,
}

family = st.radio("Any family history of mental illness?", ["No", "Yes"], horizontal=True)
features["family_history"] = 1.0 if family == "Yes" else 0.0

stress_label = "Academic pressure (0 = none … 5 = extreme)" if population == "student" \
    else "Work interference by stress (0 = never … 5 = often)"
features["stress_pressure_score"] = float(
    st.slider(stress_label, 0.0, 5.0, 2.0, 0.5)
)

if population == "student":
    c1, c2, c3 = st.columns(3)
    with c1:
        features["sleep_hours"] = float(st.slider("Sleep per night (hours)", 3.0, 12.0, 7.0, 0.5))
        features["work_study_hours"] = float(st.slider("Study/work hours per day", 0.0, 13.0, 6.0, 0.5))
    with c2:
        features["satisfaction_score"] = float(
            st.slider("Study satisfaction (0 = none … 5 = very satisfied)", 0.0, 5.0, 3.0, 0.5)
        )
        features["financial_stress"] = float(
            st.slider("Financial stress (0 = none … 5 = extreme)", 0.0, 5.0, 2.0, 0.5)
        )
    with c3:
        diet = st.select_slider(
            "Diet quality", options=["Unhealthy", "Moderate", "Healthy"], value="Moderate"
        )
        features["diet_quality"] = {"Unhealthy": 0.0, "Moderate": 1.0, "Healthy": 2.0}[diet]
        st.caption("support_available was not collected for students → left empty.")
else:
    c1, c2 = st.columns(2)
    with c1:
        features["support_available"] = float(
            {"No": 0.0, "Don't know": 0.5, "Yes": 1.0}[
                st.select_slider(
                    "Employer mental-health support",
                    options=["No", "Don't know", "Yes"],
                    value="Don't know",
                )
            ]
        )
    with c2:
        st.caption(
            "Sleep, diet, finances and satisfaction were not collected in the "
            "workplace survey → left empty (model imputes exactly as trained)."
        )

st.divider()

risk_result: dict | None = st.session_state.get("ms_risk_result")
quiz_result: dict | None = st.session_state.get("ms_quiz_result")

btn1, btn2, btn3 = st.columns([1, 1, 1])
run_risk = btn1.button("🧭 Estimate concern level", type="primary")
run_quiz = btn2.button("📋 Score questionnaires")
if btn3.button("Clear results"):
    st.session_state.pop("ms_risk_result", None)
    st.session_state.pop("ms_quiz_result", None)
    st.rerun()

if run_risk:
    try:
        risk_result = inference.predict_risk(features)
        st.session_state["ms_risk_result"] = risk_result
        if risk_result["available"]:
            fusion.record(
                "risk",
                f"concern probability {risk_result['probability']:.0%} "
                f"({risk_result['tier']} tier)",
                tone="warn" if risk_result["tier"] in {"elevated", "higher"} else "info",
            )
    except ValueError as exc:
        st.error(str(exc))

# --------------------------------------------------------------------------- #
# 2. questionnaires
# --------------------------------------------------------------------------- #
# Answers default to 0; the expander's radios (when rendered) update session state.
phq_answers = {i: int(st.session_state.get(f"ms_phq9_{i}", 0)) for i in range(1, 10)}
gad_answers = {i: int(st.session_state.get(f"ms_gad7_{i}", 0)) for i in range(1, 8)}
if phq_answers[9] > 0:
    fusion.raise_crisis("PHQ-9", "PHQ-9 item 9 (self-harm thoughts) endorsed")

with st.expander("2 · Standard questionnaires (optional) — PHQ-9 & GAD-7", expanded=False):
    st.caption(
        "How much have you been bothered by each over the **last two weeks**? "
        "PHQ-9 (depression, 0–27) and GAD-7 (anxiety, 0–21) are free, "
        "widely-used screening instruments — again, not a diagnosis."
    )

    st.markdown("**PHQ-9 — Depression**")
    for i, text in enumerate(phq9_module.PHQ9_ITEMS, start=1):
        key = f"ms_phq9_{i}"
        if key not in st.session_state:
            st.session_state[key] = 0
        value = st.radio(
            f"{i}. {text}",
            options=[0, 1, 2, 3],
            index=st.session_state[key],
            format_func=lambda v, labels=phq9_module.RESPONSE_OPTIONS: labels[v],
            key=key,
            horizontal=True,
        )
        phq_answers[i] = int(value)

    st.markdown("**GAD-7 — Anxiety**")
    for i, text in enumerate(gad7_module.GAD7_ITEMS, start=1):
        key = f"ms_gad7_{i}"
        if key not in st.session_state:
            st.session_state[key] = 0
        value = st.radio(
            f"{i}. {text}",
            options=[0, 1, 2, 3],
            index=st.session_state[key],
            format_func=lambda v, labels=phq9_module.RESPONSE_OPTIONS: labels[v],
            key=key,
            horizontal=True,
        )
        gad_answers[i] = int(value)

if run_quiz:
    if "ms_phq9_1" not in st.session_state:
        st.info("Open the questionnaire section above and answer the items first.")
    else:
        try:
            phq = phq9_module.score_phq9(phq_answers)
            gad = gad7_module.score_gad7(gad_answers)
            quiz_result = {"phq9": phq, "gad7": gad}
            st.session_state["ms_quiz_result"] = quiz_result
            tone = "crisis" if phq["crisis_triggered"] else (
                "warn" if phq["band"] in {"Moderate", "Moderately severe", "Severe"} else "info"
            )
            fusion.record(
                "questionnaire",
                f"PHQ-9 {phq['band']} ({phq['total']}/27), GAD-7 {gad['band']} ({gad['total']}/21)",
                tone=tone,
            )
            if phq["crisis_triggered"]:
                fusion.raise_crisis("PHQ-9", "PHQ-9 item 9 (self-harm thoughts) endorsed")
        except ValueError as exc:
            st.error(str(exc))

# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
if quiz_result:
    st.subheader("Questionnaire results")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**PHQ-9 (depression)**")
        widgets.kpi(f"{quiz_result['phq9']['total']} / 27", "total score")
        widgets.band_chip(quiz_result["phq9"]["band"])
        st.caption(
            "Bands: 0–4 minimal · 5–9 mild · 10–14 moderate · "
            "15–19 moderately severe · 20–27 severe."
        )
    with q2:
        st.markdown("**GAD-7 (anxiety)**")
        widgets.kpi(f"{quiz_result['gad7']['total']} / 21", "total score")
        widgets.band_chip(quiz_result["gad7"]["band"])
        st.caption("Bands: 0–4 minimal · 5–9 mild · 10–14 moderate · 15–21 severe.")
    if quiz_result["phq9"]["crisis_triggered"]:
        widgets.crisis_banner(["PHQ-9 item 9 endorsed (self-harm thoughts)"])
    st.divider()

if risk_result is not None:
    st.subheader("Estimate")
    if not risk_result.get("available"):
        widgets.unavailable(risk_result)
        st.caption(
            "The questionnaire results above are already available — they use "
            "the standard instrument scoring, not the model."
        )
    else:
        col_gauge, col_info = st.columns([3, 2], gap="large")
        with col_gauge:
            widgets.risk_gauge(risk_result["probability"], risk_result["tier"])
        with col_info:
            widgets.tier_chip(risk_result["tier"])
            st.markdown(
                f"- **Probability of elevated concern:** "
                f"**{risk_result['probability']:.0%}**\n"
                f"- **Tier cut-points:** lower `< {risk_result['thresholds']['lower']}`, "
                f"elevated `< {risk_result['thresholds']['elevated']}`, higher `≥` that\n"
                f"- **Model:** `{risk_result['model_version']}`"
            )
            st.caption(
                "A population-level statistical estimate from survey data — it "
                "has uncertainty, knows nothing we didn't ask, and cannot see "
                "your context."
            )
        if risk_result["contributions"]:
            widgets.contributions_chart(risk_result["contributions"])
        widgets.next_steps(risk_result["tier"])
        st.caption(
            "Explainability: approximate contribution of each input "
            "(Exp 8) — how each answer moved the estimate, not proof of cause."
        )

st.divider()
st.caption(
    "Questionnaire instruments: PHQ-9 (Kroenke et al., 2001) and GAD-7 "
    "(Spitzer et al., 2006), both free for clinical and non-commercial use. "
    "Lifestyle model: Experiment 6 (see Lab Results for metrics and limits)."
)
