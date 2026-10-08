"""MindSense — Home page (Section 8, page 1).

Entry point for ``streamlit run app/Home.py`` and for the AppTest smoke.
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

st.set_page_config(
    page_title="MindSense — Early, explainable mental-wellness screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.apply()
sidebar.render_sidebar()

st.title("🧠 MindSense")
st.caption("*Early, explainable mental-wellness screening.*")

st.markdown(
    """
MindSense is an **educational screening aid** that turns everyday inputs —
lifestyle, questionnaire answers, free text — into **transparent, explainable**
signals about mental-wellness risk. Every prediction comes with its confidence,
the factors behind it, and honest limits.

It is **not a diagnosis**, stores **nothing**, and is designed to start a
better conversation with a real professional — not replace one.
"""
)

if fusion.crisis_flag():
    widgets.crisis_banner(fusion.crisis_flag()["reasons"])

# --------------------------------------------------------------------------- #
# how it works
# --------------------------------------------------------------------------- #
st.subheader("How it works")
col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown("### 1️⃣ Share a little")
    st.markdown(
        "Answer a short lifestyle form, a standard questionnaire (PHQ-9/GAD-7), "
        "or paste how you've been feeling. Optionally use the face mood cue."
    )
with col2:
    st.markdown("### 2️⃣ Get explainable signals")
    st.markdown(
        "Models estimate risk, severity bands and text signals — each with its "
        "confidence, the factors that pushed it, and plain-language explanations."
    )
with col3:
    st.markdown("### 3️⃣ Decide your next step")
    st.markdown(
        "Use the tailored next steps: self-care, a check-in with someone you "
        "trust, or professional support — with crisis help always one click away."
    )

# --------------------------------------------------------------------------- #
# privacy
# --------------------------------------------------------------------------- #
st.subheader("Privacy")
st.markdown(
    """
- **Nothing is stored.** No accounts, no server-side logging, no cookies.
- Your inputs live only in this browser session and vanish when you close the tab.
- The app runs locally (or in your own container) — your data never leaves the machine.
- Datasets used for training are never shown row-by-row (licence terms respected).
"""
)

# --------------------------------------------------------------------------- #
# module status
# --------------------------------------------------------------------------- #
st.subheader("Module status")
status = inference.availability()
cols = st.columns(min(5, len(status)))
for i, info in enumerate(status.values()):
    with cols[i % len(cols)]:
        icon = "✅" if info["available"] else "⏳"
        st.markdown(f"**{icon} {info['label']}**")
        st.caption(info["detail"])
        if info["available"]:
            st.caption(f"version: `{info['version']}`")

# --------------------------------------------------------------------------- #
# session fusion summary
# --------------------------------------------------------------------------- #
if len(fusion.modules_used()) > 1:
    st.subheader("Your session so far")
    fusion.render()

# --------------------------------------------------------------------------- #
# quick links
# --------------------------------------------------------------------------- #
st.subheader("Start")
links = st.columns(3)
with links[0]:
    st.page_link("pages/1_Risk_Assessment.py", label="Risk Assessment", icon="📝")
    st.page_link("pages/2_Text_Check_in.py", label="Text Check-in", icon="💬")
with links[1]:
    st.page_link("pages/3_Prognosis_What_If.py", label="Prognosis & What-If", icon="🔮")
    st.page_link("pages/5_Population_Insights.py", label="Population Insights", icon="📊")
with links[2]:
    st.page_link("pages/6_Lab_Results.py", label="Lab Results", icon="🧪")
    st.page_link("pages/7_About_Ethics.py", label="About & Ethics", icon="⚖️")

st.divider()
st.caption(
    "MindSense · Honors Lab (AI & ML in Healthcare) mini project · "
    "datasets and methods documented in the Lab Results and About pages."
)
