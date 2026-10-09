"""Mood selfie (optional): camera/upload → emotion + mood cue.

Feature-flagged with ``ENABLE_FACE`` (on by default, ``ENABLE_FACE=0`` opts
out); when it is off, an in-app button turns the camera on for the session
without a restart. Always shows a prominent non-diagnostic label; Grad-CAM
overlay appears when the artifact provides one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

from app.components import fusion, sidebar, theme, widgets  # noqa: E402
from mindsense import inference  # noqa: E402

st.set_page_config(page_title="Mood selfie — MindSense", page_icon="📷", layout="wide")
theme.apply()
sidebar.render_sidebar()

st.title("📷 Mood selfie")
st.caption("Optional emotion cue from a face photo · runs locally on your machine")

st.markdown(
    '<div class="ms-disclaimer"><strong>NON-DIAGNOSTIC.</strong> '
    "A face model reads expressions, not people — it cannot diagnose mood "
    "disorders and its output says nothing definitive about anyone. "
    "Images are processed in memory and never saved.</div>",
    unsafe_allow_html=True,
)

if fusion.crisis_flag():
    widgets.crisis_banner(fusion.crisis_flag()["reasons"])

env_enabled = os.environ.get("ENABLE_FACE", "1") != "0"
session_enabled = bool(st.session_state.get("ms_face_enable_session"))
enabled = env_enabled or session_enabled
st.checkbox(
    "Face module active for this session",
    value=enabled,
    disabled=True,
    key="ms_face_flag",
    help="You can switch the camera on for just this visit — no restart needed.",
)

if not enabled:
    st.info(
        "The face module is switched off for privacy. You can turn it on just "
        "for this visit — images are processed in memory and never saved, and "
        "it resets when you reload the page."
    )
    if st.button(
        "🎥 Enable camera for this session",
        type="primary",
        key="ms_face_enable",
    ):
        st.session_state["ms_face_enable_session"] = True
        st.rerun()
    st.caption("The other pages work fully without it.")
    st.stop()

source = st.radio("Image source", ["📷 Camera", "🖼 Upload a photo"], horizontal=True)
image_bytes: bytes | None = None
if source == "📷 Camera":
    if st.session_state.get("ms_cam_active"):
        captured = st.camera_input("Look at the camera")
        if captured is not None:
            image_bytes = captured.getvalue()
        if st.button("⏹ Stop camera", key="ms_cam_stop"):
            st.session_state["ms_cam_active"] = False
            st.rerun()
    else:
        st.caption(
            "Press start to switch on your camera. Your browser will ask for "
            "permission; nothing is recorded and images are never saved."
        )
        if st.button("🎥 Start camera", type="primary", key="ms_cam_start"):
            st.session_state["ms_cam_active"] = True
            st.rerun()
else:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        image_bytes = uploaded.getvalue()

if image_bytes:
    try:
        result = inference.analyze_face(image_bytes, enabled=enabled)
    except ValueError as exc:
        st.error(str(exc))
        result = None

    if result is not None:
        if not result.get("available"):
            widgets.unavailable(result)
        else:
            left, right = st.columns([2, 3])
            with left:
                st.image(image_bytes, caption="Your image (not saved)", width="stretch")
            with right:
                widgets.kpi(result["top_emotion"], f"top emotion · mood cue: {result['mood_cue']}")
                widgets.probability_bars(result["probabilities"], title="Emotion probabilities")
                st.caption(f"model: `{result['model_version']}`")
            if result.get("gradcam") is not None:
                st.markdown("**Grad-CAM overlay** — where the model looked:")
                st.image(result["gradcam"], width="stretch")
            else:
                st.caption(
                    "Grad-CAM overlay becomes available when the photo model "
                    "artifact includes it (see Models & data)."
                )
            fusion.record(
                "face",
                f"emotion `{result['top_emotion']}` (cue: {result['mood_cue']})",
                tone="info",
            )

st.divider()
st.caption(
    "FER-2013 — 7-class emotion recognition; per-class metrics and "
    "Grad-CAM examples on **Models & data**."
)
