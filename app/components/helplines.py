"""Helplines from ``config/helplines.yaml`` — rendered in the sidebar dialog, crisis banners and About page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from mindsense.utils.io import load_helplines


def regions() -> dict[str, Any]:
    return load_helplines()["regions"]


def region_names() -> list[str]:
    return list(regions())


def region_lines(region: str) -> list[dict[str, Any]]:
    """Crisis-line entries for ``region`` (``{lines: [...]}`` in the YAML)."""
    entry = regions().get(region, {})
    if isinstance(entry, dict):
        return list(entry.get("lines", []))
    return list(entry or [])


def default_region() -> str:
    cfg = load_helplines()
    return str(cfg.get("default_region", region_names()[0]))


def current_region() -> str:
    """Selected region (session-scoped; defaults from config)."""
    name = st.session_state.get("ms_region")
    if name not in regions():
        name = default_region()
        st.session_state["ms_region"] = name
    return name


def render_lines(region: str | None = None) -> None:
    """Render the crisis lines for ``region`` (current selection by default)."""
    region = region or current_region()
    lines = region_lines(region)
    if not lines:
        st.warning(f"No helpline entries configured for {region}. See findahelpline.com.")
        return
    for line in lines:
        number = line.get("number")
        url = line.get("url")
        title = f"**{line['name']}**"
        parts = [title]
        if number:
            parts.append(f"☎️ {number}")
        st.markdown(" — ".join(parts))
        if line.get("note"):
            st.caption(line["note"])
        if url:
            st.link_button("Open website", url)
        st.divider()
