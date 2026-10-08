"""MindSense — calm, customer-facing home page.

Entry point for ``streamlit run app/Home.py`` and for the AppTest smoke.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

from app.components import fusion, helplines, nav, sidebar, theme, widgets  # noqa: E402

st.set_page_config(
    page_title="MindSense — a calmer way to check in with yourself",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.apply()
sidebar.render_sidebar()

# --------------------------------------------------------------------------- #
# hero
# --------------------------------------------------------------------------- #
theme.hero(
    "A calmer way to check in with yourself.",
    "Two quiet minutes. A few honest questions. Clear, gentle feedback you "
    "can actually act on — private by design, with a little AI kindness at "
    "the end.",
    gradient_word="calmer",
)

cta1, cta2, _pad = st.columns([1.2, 1.2, 2])
with cta1:
    nav.page_link("pages/1_Check_In.py", label="Start your check-in", icon="📝")
with cta2:
    nav.page_link("pages/2_Talk_it_Out.py", label="Talk it out instead", icon="💬")

if fusion.crisis_flag():
    widgets.crisis_banner(fusion.crisis_flag()["reasons"])

# --------------------------------------------------------------------------- #
# three promise cards
# --------------------------------------------------------------------------- #
cards = st.columns(3, gap="medium")
with cards[0]:
    st.markdown(
        theme.card(
            "Fast",
            "About 2 minutes",
            "A short lifestyle form plus two standard questionnaires "
            "(PHQ-9 &amp; GAD-7) — the same free instruments clinics use.",
            icon="⏳",
        ),
        unsafe_allow_html=True,
    )
with cards[1]:
    st.markdown(
        theme.card(
            "Private",
            "Nothing is stored",
            "No accounts, no logging. Answers live only in this tab and vanish "
            "when you close it. AI suggestions see numbers, never your identity.",
            icon="🔒",
        ),
        unsafe_allow_html=True,
    )
with cards[2]:
    st.markdown(
        theme.card(
            "Kind",
            "Gentle, useful feedback",
            "Plain-language results, honest gaps, small suggested steps — and "
            "real helplines one tap away, always.",
            icon="🌿",
        ),
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------- #
# quick mood log (session-only, feeds the Mood patterns page)
# --------------------------------------------------------------------------- #
mood_log: list[int] = st.session_state.setdefault("ms_mood_log", [])
log_box = st.container(border=True)
with log_box:
    st.markdown("### 🌤️ One-second mood check")
    st.caption("No form, no score — just a tap. Stored only in this tab.")
    mood_cols = st.columns([3, 1])
    with mood_cols[0]:
        mood = st.select_slider(
            "Right now I feel…",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda v: {
                1: "😞 low",
                2: "🙁 unsettled",
                3: "😐 steady",
                4: "🙂 good",
                5: "😄 bright",
            }[v],
            key="ms_home_mood",
        )
    with mood_cols[1]:
        if st.button("Log it", type="primary", width="stretch", key="ms_home_mood_btn"):
            mood_log.append(int(mood))
            st.toast("Noted — thank you for checking in 💚")
    if mood_log:
        st.caption(
            f"{len(mood_log)} mood point(s) this session · see the trend on **Mood patterns**."
        )

st.write("")

# --------------------------------------------------------------------------- #
# safety strip
# --------------------------------------------------------------------------- #
safety = st.container(border=True)
with safety:
    st.markdown("### 🆘 Need help right now?")
    st.caption(
        "If you ever feel unsafe or at risk, don't wait — talk to someone "
        "today. Verified helplines are one click away."
    )
    if st.button("🆘 Help now", type="primary", key="ms_home_help"):
        st.session_state["ms_help_open"] = True
        st.rerun()
    helpline_box, _ = st.columns([4, 1])
    with helpline_box:
        helplines.render_lines()

# --------------------------------------------------------------------------- #
# session recap
# --------------------------------------------------------------------------- #
if len(fusion.modules_used()) > 1:
    st.subheader("Your session so far")
    fusion.render()

st.write("")
st.divider()
st.caption(
    "MindSense · educational screening aid · not a medical diagnosis · "
    "methods & sources on **Care & safety** and **Models & data**."
)
