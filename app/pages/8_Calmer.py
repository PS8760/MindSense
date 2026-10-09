"""Calmer — a calm companion chatbot + mood-lifting content.

The chatbot shows a short "thinking" step before its answer (the model returns
both). Everything is optional-AI: with no Groq key it falls back to gentle,
hand-written copy and never breaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

from app.components import ai, sidebar, theme, widgets  # noqa: E402

st.set_page_config(page_title="Calmer — MindSense", page_icon="🫧", layout="wide")
theme.apply()
sidebar.render_sidebar()

st.title("🫧 Calmer")
st.caption(
    "A quiet space to breathe and be heard. Calmer is a supportive companion — "
    "not a therapist or a diagnosis. If you're in danger, use **Help now**."
)

# --------------------------------------------------------------------------- #
# mood-lifting content
# --------------------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("### 🌤️ A little lift, right now")
    st.caption("Pick how you feel and ask for something to lift your mood.")
    mood_col, btn_col = st.columns([3, 1])
    with mood_col:
        mood = st.select_slider(
            "My mood right now",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda v: {
                1: "😞 low",
                2: "🙁 unsettled",
                3: "😐 steady",
                4: "🙂 good",
                5: "😄 bright",
            }[v],
            key="ms_calmer_mood",
        )
    with btn_col:
        st.write("")
        ask = st.button("Lift my mood", type="primary", width="stretch")

    if ask:
        with st.spinner("Finding something gentle for you…"):
            st.session_state["ms_calmer_uplift"] = ai.uplift_content(int(mood))

    uplift = st.session_state.get("ms_calmer_uplift")
    if uplift:
        cards = st.columns(2, gap="medium")
        for i, item in enumerate(uplift["items"]):
            with cards[i % 2]:
                kind = item.get("kind") or "Idea"
                st.markdown(f"**{kind} · {item['title']}**")
                st.caption(item["body"])
        if uplift.get("source") == "groq":
            st.caption("✨ Fresh ideas from your AI assistant.")
        else:
            st.caption("📴 Offline ideas — always here, written by us.")

st.write("")

# --------------------------------------------------------------------------- #
# Calmer chat
# --------------------------------------------------------------------------- #
st.markdown("### 💬 Talk with Calmer")

chat_col, clear_col = st.columns([4, 1])
with clear_col:
    if st.button("Clear chat", width="stretch", key="ms_calmer_clear"):
        st.session_state["ms_calmer"] = []
        st.rerun()

history: list[dict[str, str]] = st.session_state.setdefault("ms_calmer", [])
if not history:
    st.caption("Say hello — or share whatever is on your mind. Nothing is stored.")

for message in history:
    avatar = "🫧" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input("Tell Calmer how you're feeling…")
if prompt:
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🫧"):
        with st.status("Calmer is thinking…", expanded=True) as status:
            result = ai.calmer_reply(history)
            st.write(result.get("thinking") or "Listening carefully to what you need.")
            status.update(label="Calmer", state="complete", expanded=False)
        st.markdown(result["reply"])
        if result.get("source") == "groq":
            st.caption("✨ Calmer is an AI companion — supportive, not a diagnosis.")
        else:
            st.caption("📴 Offline reply — a steady voice is always here.")
    history.append({"role": "assistant", "content": result["reply"]})
    crisis = result.get("crisis") or {}
    if crisis.get("triggered"):
        widgets.crisis_banner(crisis.get("reasons", []))

st.divider()
st.caption(
    "Calmer offers comfort and small coping steps only. It cannot diagnose or "
    "treat anything. For anything serious, please reach out to a professional "
    "or a helpline (**Help now**)."
)
