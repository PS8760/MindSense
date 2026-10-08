"""Mood selfie (optional): camera/upload → emotion + mood cue.

Feature-flagged with ``ENABLE_FACE=1`` (Section 8, page 5). Always shows a
prominent non-diagnostic label; Grad-CAM overlay appears when the artifact
provides one.
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

enabled = os.environ.get("ENABLE_FACE", "0") == "1"
st.checkbox(
    "Enable face module for this session (requires ENABLE_FACE=1 at launch)",
    value=enabled,
    disabled=True,
    key="ms_face_flag",
    help="Start the app with ENABLE_FACE=1 to switch this on.",
)

if not enabled:
    st.info(
        "The face module is **off by default** (privacy and cold-start budget). "
        "To try it, relaunch with:\n\n"
        "```bash\nENABLE_FACE=1 streamlit run app/Home.py\n```",
    )
    st.caption(
        "The other six pages work fully without it. Trained weights also require ``make train``."
    )
    st.stop()

source = st.radio("Image source", ["📷 Camera", "🖼 Upload a photo"], horizontal=True)
image_bytes: bytes | None = None
if source == "📷 Camera":
    captured = st.camera_input("Look at the camera")
    if captured is not None:
        image_bytes = captured.getvalue()
else:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        image_bytes = uploaded.getvalue()

if image_bytes:
    try:
        result = inference.analyze_face(image_bytes)
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
