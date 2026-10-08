"""Persistent sidebar: non-dismissible disclaimer, Help now, region, fusion.

Every page calls :func:`render_sidebar` once (after ``set_page_config``).
"""

from __future__ import annotations

import streamlit as st

from app.components import fusion, helplines, theme, widgets

DISCLAIMER = (
    "⚠️ **Not a medical diagnosis. If you are in distress, seek help.**\n\n"
    "MindSense is an educational screening aid. It cannot diagnose, treat or "
    "replace a health professional. Nothing you type here is stored or sent "
    "anywhere — everything runs in this browser session."
)


@st.dialog("Help now", width="large")
def _help_dialog() -> None:
    st.markdown("### 🆘 Help is available right now")
    st.warning(
        "If you may act on thoughts of harming yourself, or feel unsafe, "
        "contact emergency services or a crisis line now — don't wait.",
        icon="🚨",
    )
    helplines.render_lines()
    st.caption(
        "Numbers were verified on the date shown in config/helplines.yaml — "
        "re-check before relying on them in an emergency."
    )
    if st.button("Close", type="primary"):
        st.session_state.pop("ms_help_open", None)
        st.rerun()


def render_sidebar() -> None:
    """Render the persistent sidebar (disclaimer → help → region → status)."""
    with st.sidebar:
        st.markdown("## 🧠 MindSense")
        st.caption("Early, explainable mental-wellness screening.")
        st.markdown(
            f'<div class="ms-disclaimer">{DISCLAIMER.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        crisis = fusion.crisis_flag()
        if crisis:
            widgets.crisis_banner(crisis["reasons"])

        if st.button("🆘 Help now", type="primary", width="stretch", key="ms_help_btn"):
            st.session_state["ms_help_open"] = True
        if st.session_state.get("ms_help_open"):
            _help_dialog()

        st.selectbox(
            "Region for helplines",
            helplines.region_names(),
            key="ms_region",
            help="Used by Help now, the crisis banner and the About page.",
        )

        dark_now = theme.current_mode() == "Dark"
        target = "Light" if dark_now else "Dark"
        if st.button(
            "☀️ Light mode" if dark_now else "🌙 Dark mode",
            key="ms_theme_btn",
            width="stretch",
        ):
            st.session_state["ms_theme"] = target
            st.rerun()
        if theme.current_mode() == "System":
            st.caption("Following your device — tap the button to pick one.")

        used = fusion.modules_used()
        if used:
            with st.expander("Session summary", expanded=False):
                fusion.render(compact=True)

        st.divider()
        st.caption(
            "Run entirely on your machine · no accounts · no logging · "
            "data leaves nothing behind when you close the tab."
        )


def ensure_region() -> None:
    """Initialize the region key before any render (idempotent)."""
    helplines.current_region()
