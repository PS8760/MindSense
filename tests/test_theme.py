"""Theme tests: both palettes render, variables switch, card/hero escape text."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.components import theme

REPO = Path(__file__).resolve().parents[1]


def test_palettes_differ_on_key_colors() -> None:
    light, dark = theme._palette(False), theme._palette(True)
    assert light["ms-bg"] != dark["ms-bg"]
    assert light["ms-ink"] != dark["ms-ink"]
    assert light["ms-primary"] != dark["ms-primary"]


def test_button_ink_readable_in_both_modes() -> None:
    light, dark = theme._palette(False), theme._palette(True)
    assert light["ms-btn-ink"] == "#2E8B7A"  # visible teal on white button
    assert dark["ms-btn-ink"] == "#06231C"  # dark text on mint button


def test_vars_block_renders_css_custom_properties() -> None:
    block = theme._vars_block({"ms-ink": "#E7EEEC", "ms-bg": "#0E1616"})
    assert "--ms-ink: #E7EEEC;" in block
    assert "--ms-bg: #0E1616;" in block


def test_light_css_matches_light_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme, "current_mode", lambda: "Light")
    css = theme._css()
    assert theme._palette(False)["ms-bg"] in css
    assert "prefers-color-scheme" not in css


def test_dark_css_matches_dark_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme, "current_mode", lambda: "Dark")
    css = theme._css()
    assert theme._palette(True)["ms-bg"] in css
    assert "prefers-color-scheme" not in css


def test_system_css_guards_dark_under_media_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theme, "current_mode", lambda: "System")
    css = theme._css()
    assert theme._palette(False)["ms-bg"] in css
    assert theme._palette(True)["ms-bg"] in css
    assert "prefers-color-scheme" in css


def test_card_escapes_kicker_and_title() -> None:
    html = theme.card("k<e>", "<b>Title</b>", "Body &amp; more")
    assert "<b>Title</b>" not in html
    assert "&lt;b&gt;Title&lt;/b&gt;" in html
    assert "k&lt;e&gt;" in html
    assert "Body &amp; more" in html  # body is intentionally raw HTML


def test_theme_button_toggles_and_round_trips() -> None:
    at = AppTest.from_file(str(REPO / "app" / "Home.py"), default_timeout=60)
    at.run()
    assert not at.exception
    button = [b for b in at.button if b.key == "ms_theme_btn"]
    assert button, "theme toggle button should exist"
    first_label = button[0].label
    button[0].click()
    at.run()
    assert not at.exception
    assert first_label == "🌙 Dark mode"  # System/light default → offers dark
    second = [b for b in at.button if b.key == "ms_theme_btn"]
    assert second and second[0].label == "☀️ Light mode"  # now in dark → offers light
    second[0].click()
    at.run()
    assert not at.exception
    third = [b for b in at.button if b.key == "ms_theme_btn"]
    assert third and third[0].label == "🌙 Dark mode"  # back to light
