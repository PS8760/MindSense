"""Navigation helper: ``st.page_link`` that stays graceful in bare/test mode.

Real production boots from ``app/Home.py`` with a ``pages/`` directory present,
so every link resolves. Standalone AppTest entrypoints (a page opened as the
main script) have no page registry and would otherwise raise; in that case we
render a matching label instead so the UI never hard-crashes.
"""

from __future__ import annotations

import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError


def page_link(
    page: str, *, label: str | None = None, icon: str = "", disabled: bool = False
) -> None:
    try:
        st.page_link(page, label=label, icon=icon, disabled=disabled)
    except StreamlitPageNotFoundError:
        if label and not disabled:
            st.markdown(f"{icon} **{label}**")
