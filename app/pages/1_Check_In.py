"""Your check-in: lifestyle form + PHQ-9/GAD-7 → results, gaps, suggestions.

All model calls go through :mod:`mindsense.inference`. PHQ-9 item 9 > 0
raises the crisis banner immediately. Rule-based gaps (always available) and
Groq-powered suggestions complete the "prediction → awareness → action" flow.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

from app.components import ai, fusion, nav, sidebar, theme, widgets  # noqa: E402
from mindsense import inference  # noqa: E402
from mindsense.screening import gad7 as gad7_module  # noqa: E402
from mindsense.screening import phq9 as phq9_module  # noqa: E402

st.set_page_config(page_title="Your check-in — MindSense", page_icon="📝", layout="wide")
theme.apply()
sidebar.render_sidebar()

risk_result: dict | None = st.session_state.get("ms_risk_result")
quiz_result: dict | None = st.session_state.get("ms_quiz_result")
has_results = risk_result is not None or quiz_result is not None

theme.hero(
    "Your check-in",
    "A few honest questions — then a clear, kind picture of where you stand "
    "and small steps that fit your answers.",
)
theme.steps(
    "1 · About you",
    "2 · How you've been feeling",
    "3 · Your results",
    active=2 if has_results else 0,
)

if fusion.crisis_flag():
    widgets.crisis_banner(fusion.crisis_flag()["reasons"])

# --------------------------------------------------------------------------- #
# step 1 — lifestyle form
# --------------------------------------------------------------------------- #
st.subheader("1 · A little about you")

col_pop, col_demo = st.columns([1, 2])
with col_pop:
    population = st.radio(
        "I am a…",
        ["student", "professional"],
        format_func=lambda p: "Student" if p == "student" else "Working professional",
        horizontal=True,
        help="Two source populations were combined; each question is asked "
        "only where the original survey actually asked it.",
    )
with col_demo:
    age = st.number_input("Age", min_value=5, max_value=120, value=22, step=1)
    gender = st.selectbox("Gender", ["male", "female", "other"], index=0)

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

stress_label = (
    "Academic pressure (0 = none … 5 = extreme)"
    if population == "student"
    else "Work interference by stress (0 = never … 5 = often)"
)
features["stress_pressure_score"] = float(st.slider(stress_label, 0.0, 5.0, 2.0, 0.5))

if population == "student":
    c1, c2, c3 = st.columns(3)
    with c1:
        features["sleep_hours"] = float(st.slider("Sleep per night (hours)", 3.0, 12.0, 7.0, 0.5))
        features["work_study_hours"] = float(
            st.slider("Study/work hours per day", 0.0, 13.0, 6.0, 0.5)
        )
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
            "Sleep, diet, finances and satisfaction weren't part of the "
            "workplace survey — left empty rather than guessed."
        )

# --------------------------------------------------------------------------- #
# step 2 — questionnaires
# --------------------------------------------------------------------------- #
with st.expander("2 · How you've been feeling — PHQ-9 & GAD-7 (optional)", expanded=False):
    st.caption(
        "Over the **last two weeks**, how often have you been bothered by each? "
        "PHQ-9 (low mood) and GAD-7 (anxiety) are free, standard screening "
        "instruments — signals, not diagnoses."
    )

    st.markdown("**PHQ-9 — low mood**")
    for i, text in enumerate(phq9_module.PHQ9_ITEMS, start=1):
        key = f"ms_phq9_{i}"
        value = st.radio(
            f"{i}. {text}",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get(key, 0)),
            format_func=lambda v, labels=phq9_module.RESPONSE_OPTIONS: labels[v],
            key=key,
            horizontal=True,
        )
        st.session_state.setdefault(key, int(value))

    st.markdown("**GAD-7 — anxiety**")
    for i, text in enumerate(gad7_module.GAD7_ITEMS, start=1):
        key = f"ms_gad7_{i}"
        value = st.radio(
            f"{i}. {text}",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get(key, 0)),
            format_func=lambda v, labels=phq9_module.RESPONSE_OPTIONS: labels[v],
            key=key,
            horizontal=True,
        )
        st.session_state.setdefault(key, int(value))

# --------------------------------------------------------------------------- #
# CTA
# --------------------------------------------------------------------------- #
cta1, cta2, _pad = st.columns([1.4, 1, 2])
run_all = cta1.button("See my results", type="primary", width="stretch")
if cta2.button("Start over", width="stretch"):
    for k in list(st.session_state):
        if k.startswith(("ms_risk_result", "ms_quiz_result", "ms_phq9_", "ms_gad7_")):
            st.session_state.pop(k, None)
    st.rerun()

if run_all:
    try:
        risk_result = inference.predict_risk(features)
        st.session_state["ms_risk_result"] = risk_result
        if risk_result["available"]:
            fusion.record(
                "check-in",
                f"concern estimate {risk_result['probability']:.0%} ({risk_result['tier']} tier)",
                tone="warn" if risk_result["tier"] in {"elevated", "higher"} else "info",
            )
    except ValueError as exc:
        st.error(str(exc))

    phq_answers = {i: int(st.session_state.get(f"ms_phq9_{i}", 0)) for i in range(1, 10)}
    gad_answers = {i: int(st.session_state.get(f"ms_gad7_{i}", 0)) for i in range(1, 8)}
    if phq_answers[9] > 0:
        fusion.raise_crisis("PHQ-9", "PHQ-9 item 9 (self-harm thoughts) endorsed")
    try:
        phq = phq9_module.score_phq9(phq_answers)
        gad = gad7_module.score_gad7(gad_answers)
        quiz_result = {"phq9": phq, "gad7": gad}
        st.session_state["ms_quiz_result"] = quiz_result
        tone = (
            "crisis"
            if phq["crisis_triggered"]
            else ("warn" if phq["band"] in {"Moderate", "Moderately severe", "Severe"} else "info")
        )
        fusion.record(
            "check-in",
            f"PHQ-9 {phq['band']} ({phq['total']}/27), GAD-7 {gad['band']} ({gad['total']}/21)",
            tone=tone,
        )
        if phq["crisis_triggered"]:
            fusion.raise_crisis("PHQ-9", "PHQ-9 item 9 (self-harm thoughts) endorsed")
    except ValueError as exc:
        st.error(str(exc))
    st.rerun()

if not has_results and not (risk_result or quiz_result):
    st.info(
        "Answer what you can above, then press **See my results**. "
        "Questionnaires are optional — the estimate works without them."
    )
    st.stop()

risk_result = st.session_state.get("ms_risk_result")
quiz_result = st.session_state.get("ms_quiz_result")

# --------------------------------------------------------------------------- #
# step 3 — results
# --------------------------------------------------------------------------- #
st.subheader("3 · Your results")

top = st.columns([3, 2], gap="large")
with top[0]:
    if risk_result and risk_result.get("available"):
        widgets.risk_gauge(risk_result["probability"], risk_result["tier"])
    elif risk_result is not None:
        widgets.unavailable(risk_result)
        st.caption("Questionnaire scores below don't need the model — they use official scoring.")
    else:
        st.caption("Lifestyle estimate skipped — press **See my results** to include it.")
with top[1]:
    if quiz_result:
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**Low mood (PHQ-9)**")
            widgets.kpi(f"{quiz_result['phq9']['total']} / 27", "score")
            widgets.band_chip(quiz_result["phq9"]["band"])
        with q2:
            st.markdown("**Anxiety (GAD-7)**")
            widgets.kpi(f"{quiz_result['gad7']['total']} / 21", "score")
            widgets.band_chip(quiz_result["gad7"]["band"])
        if all(int(st.session_state.get(f"ms_phq9_{i}", 0)) == 0 for i in range(1, 10)) and all(
            int(st.session_state.get(f"ms_gad7_{i}", 0)) == 0 for i in range(1, 8)
        ):
            st.caption("Questionnaire items were left at “Not at all” — scores of 0.")
        if quiz_result["phq9"]["crisis_triggered"]:
            widgets.crisis_banner(["PHQ-9 item 9 endorsed (self-harm thoughts)"])
    else:
        st.caption("Questionnaires skipped — mood scores appear here once answered.")

if risk_result and risk_result.get("available"):
    if risk_result["contributions"]:
        widgets.contributions_chart(risk_result["contributions"])
        st.caption(
            "How each answer moved the estimate — approximate, associational, never proof of cause."
        )
    widgets.next_steps(risk_result["tier"])

st.divider()

# --------------------------------------------------------------------------- #
# gaps (rule-based, always available)
# --------------------------------------------------------------------------- #
gaps = ai.compute_gaps(features, quiz_result)
st.subheader("What stands out")

half = len(gaps) // 2 + len(gaps) % 2
left, right = st.columns(2, gap="medium")
for index, gap in enumerate(gaps):
    column = left if index < half else right
    with column:
        st.markdown(
            theme.card("Notice", gap["title"], escape(gap["detail"]), icon=gap["icon"]),
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------------------------------- #
# suggestions (Groq, offline fallback)
# --------------------------------------------------------------------------- #
st.subheader("A gentle next step")
context: dict[str, object] = {
    "population": population,
    "age_bucket": f"{(int(age) // 10) * 10}s",
    "sleep_hours": features.get("sleep_hours"),
    "stress": features.get("stress_pressure_score"),
    "study_work_hours": features.get("work_study_hours"),
    "financial_stress": features.get("financial_stress"),
    "satisfaction": features.get("satisfaction_score"),
    "diet": features.get("diet_quality"),
    "support": features.get("support_available"),
    "family_history": bool(features.get("family_history")),
    "phq9": (
        {"total": quiz_result["phq9"]["total"], "band": quiz_result["phq9"]["band"]}
        if quiz_result
        else None
    ),
    "gad7": (
        {"total": quiz_result["gad7"]["total"], "band": quiz_result["gad7"]["band"]}
        if quiz_result
        else None
    ),
    "risk_tier": risk_result.get("tier") if risk_result and risk_result.get("available") else None,
    "gaps": [g["title"] for g in gaps],
}
suggestion_box = st.container(border=True)
with suggestion_box:
    ai.render_suggestions(context)

st.divider()

next1, next2, next3 = st.columns(3)
with next1:
    nav.page_link("pages/2_Talk_it_Out.py", label="Talk it out", icon="💬")
with next2:
    nav.page_link("pages/3_Mood_Patterns.py", label="Mood patterns", icon="🌙")
with next3:
    nav.page_link("pages/7_Care_and_Safety.py", label="Care & safety", icon="⚖️")

st.caption(
    "PHQ-9 (Kroenke et al., 2001) & GAD-7 (Spitzer et al., 2006) — free, "
    "standard instruments · results are screening signals, never a diagnosis."
)
