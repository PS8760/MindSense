"""Calm, minimal visual theme for MindSense — light and dark.

Everything is driven by CSS custom properties, so the same stylesheet renders
either palette. Use |mode| in the sidebar (System / Light / Dark); "System"
follows ``prefers-color-scheme`` via a media query. Native Streamlit chrome
that cannot be themed at runtime (buttons, inputs, dialogs, dataframes) is
re-coloured from the ``--ms-*`` variables here.
"""

from __future__ import annotations

from html import escape

import streamlit as st

_THEME_KEYS = ("System", "Light", "Dark")


def _palette(dark: bool) -> dict[str, str]:
    # Calm "wellbeing" scheme: soothing teal-green (nature, growth, calm) with a
    # soft lavender/periwinkle secondary (serenity). Muted, low-stimulus whole
    # tones in dark; airy mist + crisp white in light.
    if not dark:
        return {
            "ms-bg": "#F2F7F5",
            "ms-surface": "#FFFFFF",
            "ms-line": "#DDE8E5",
            "ms-primary": "#2E8B7A",
            "ms-primary-dark": "#256F62",
            "ms-primary-soft": "#DFF0EC",
            "ms-primary-contrast": "#FFFFFF",
            "ms-blue": "#8A7FC7",
            "ms-blue-soft": "#EEECFA",
            "ms-ink": "#22333C",
            "ms-body": "#3C4E58",
            "ms-muted": "#6B7F8A",
            "ms-amber": "#B07E2E",
            "ms-amber-soft": "#FAF1DE",
            "ms-rose": "#B4584E",
            "ms-rose-soft": "#F8EAE7",
            "ms-rose-ink": "#7C322B",
            "ms-glow1": "#E6F2EF",
            "ms-glow2": "#EEEDF9",
            "ms-side1": "#EEF5F2",
            "ms-side2": "#EFEDF9",
            "ms-grad-a": "#2E8B7A",
            "ms-grad-b": "#8A7FC7",
            "ms-btn-bg": "#FFFFFF",
            "ms-btn-ink": "#2E8B7A",
            "ms-shadow": "0 1px 2px rgba(34,51,60,.05), 0 10px 30px rgba(34,51,60,.06)",
        }
    return {
        "ms-bg": "#0D1516",
        "ms-surface": "#152120",
        "ms-line": "#243331",
        "ms-primary": "#8FD6C2",
        "ms-primary-dark": "#6FC0AB",
        "ms-primary-soft": "#12382F",
        "ms-primary-contrast": "#06231C",
        "ms-blue": "#B4AEE8",
        "ms-blue-soft": "#252450",
        "ms-ink": "#E9F1EF",
        "ms-body": "#B9CAC6",
        "ms-muted": "#8FA3A0",
        "ms-amber": "#E3B56B",
        "ms-amber-soft": "#3A2F17",
        "ms-rose": "#E08A80",
        "ms-rose-soft": "#3A2522",
        "ms-rose-ink": "#F5CCC6",
        "ms-glow1": "#0C211D",
        "ms-glow2": "#161528",
        "ms-side1": "#0D1B19",
        "ms-side2": "#131226",
        "ms-grad-a": "#8FD6C2",
        "ms-grad-b": "#B4AEE8",
        "ms-btn-bg": "#8FD6C2",
        "ms-btn-ink": "#06231C",
        "ms-shadow": "0 1px 2px rgba(0,0,0,.35), 0 10px 30px rgba(0,0,0,.28)",
    }


def _vars_block(entries: dict[str, str]) -> str:
    lines = "".join(f"  --{k}: {v};\n" for k, v in entries.items())
    return f":root {{\n{lines}}}"


def _root_css() -> str:
    """The ``:root`` variable block for the active theme mode."""
    mode = current_mode()
    if mode == "Dark":
        return _vars_block(_palette(True))
    if mode == "Light":
        return _vars_block(_palette(False))
    return (
        _vars_block(_palette(False))
        + "\n@media (prefers-color-scheme: dark) {\n"
        + _vars_block(_palette(True))
        + "\n}"
    )


def current_mode() -> str:
    """Active appearance mode, defaulting to ``System``."""
    mode = st.session_state.get("ms_theme", "System")
    return mode if mode in _THEME_KEYS else "System"


def theme_options() -> tuple[str, ...]:
    return _THEME_KEYS


_LAYOUT = """
/* ---------- base ---------- */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
               Roboto, "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 460px at 12% -8%, var(--ms-glow1) 0%, transparent 55%),
    radial-gradient(900px 440px at 92% -4%, var(--ms-glow2) 0%, transparent 52%),
    var(--ms-bg);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

h1, h2, h3, h4 { color: var(--ms-ink); letter-spacing: -0.02em; font-weight: 700; }
h1 { font-size: 2.15rem; }
h2 { font-size: 1.45rem; }
h3 { font-size: 1.12rem; }
p, li { line-height: 1.65; }
[data-testid="stMarkdownContainer"] p { color: var(--ms-body); }
[data-testid="stCaptionContainer"] { color: var(--ms-muted); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--ms-side1) 0%, var(--ms-side2) 100%);
  border-right: 1px solid var(--ms-line);
}

/* ---------- hero & cards ---------- */
.ms-hero { padding: 2.2rem 0 0.4rem; animation: msFadeUp .55s ease both; }
.ms-hero h1 { color: var(--ms-ink); font-size: clamp(2.1rem, 4vw, 3rem); line-height: 1.12; margin-bottom: .4rem; }
.ms-hero .ms-grad {
  background: linear-gradient(92deg, var(--ms-grad-a) 10%, var(--ms-grad-b) 90%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.ms-hero p { color: var(--ms-muted); font-size: 1.08rem; max-width: 58ch; }
.ms-card {
  background: var(--ms-surface);
  border: 1px solid var(--ms-line);
  border-radius: var(--ms-radius, 16px);
  padding: 1.25rem 1.35rem;
  box-shadow: var(--ms-shadow);
  height: 100%;
  animation: msFadeUp .5s ease both;
}
.ms-card h4 { margin: .15rem 0 .4rem; font-size: 1.05rem; }
.ms-card p { color: var(--ms-muted); font-size: .95rem; margin: 0; }
.ms-kicker {
  display: inline-block; font-size: .74rem; font-weight: 700;
  letter-spacing: .09em; text-transform: uppercase;
  color: var(--ms-primary); background: var(--ms-primary-soft);
  padding: .18rem .55rem; border-radius: 999px;
}
.ms-emoji { font-size: 1.5rem; }
@keyframes msFadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.ms-fade { animation: msFadeUp .5s ease both; }

/* ---------- stepper ---------- */
.ms-steps { display: flex; gap: .55rem; flex-wrap: wrap; margin: .3rem 0 1.1rem; }
.ms-step {
  display: inline-flex; align-items: center; gap: .5rem;
  background: var(--ms-surface); border: 1px solid var(--ms-line);
  border-radius: 999px; padding: .42rem .95rem;
  font-size: .9rem; font-weight: 600; color: var(--ms-muted);
}
.ms-step.is-active { border-color: var(--ms-primary); color: var(--ms-primary); background: var(--ms-primary-soft); }
.ms-step.is-done { color: var(--ms-primary-dark); }

/* ---------- native widgets re-coloured from vars ---------- */
div[data-testid="stButton"] button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--ms-btn-bg) !important;
  border: 1px solid var(--ms-line) !important;
  color: var(--ms-btn-ink) !important;
  border-radius: 12px; font-weight: 600;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
div[data-testid="stButton"] button:hover { transform: translateY(-1px); border-color: var(--ms-primary) !important; }
div[data-testid="stTooltipHoverTarget"] > [data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  border-color: transparent !important;
  color: var(--ms-muted) !important;
}
div[data-testid="stTooltipHoverTarget"] > [data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stTooltipHoverTarget"] > [data-testid="stBaseButton-secondary"]:focus {
  background: transparent !important;
  color: var(--ms-muted) !important;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:active,
[data-testid="stBaseButton-secondary"]:focus,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:active,
[data-testid="stBaseButton-secondaryFormSubmit"]:focus {
  background: var(--ms-btn-bg) !important;
  color: var(--ms-btn-ink) !important;
  border-color: var(--ms-primary) !important;
}
[data-testid="stBaseButton-secondary"]:focus-visible {
  box-shadow: 0 0 0 3px var(--ms-primary-soft), 0 0 0 1.5px var(--ms-primary) !important;
}
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"],
[kind="primary"] {
  background: var(--ms-primary) !important;
  border-color: var(--ms-primary) !important;
  color: var(--ms-primary-contrast) !important;
}
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:active,
[data-testid="stBaseButton-primary"]:focus,
[data-testid="stBaseButton-primary"]:focus-visible,
[data-testid="stBaseButton-primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:active,
[data-testid="stBaseButton-primaryFormSubmit"]:focus {
  background: var(--ms-primary-dark) !important;
  color: var(--ms-primary-contrast) !important;
}
[data-testid="stBaseButton-primary"]:focus-visible {
  box-shadow: 0 0 0 3px var(--ms-primary-soft), 0 0 0 1.5px var(--ms-primary) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"],
[data-testid="stFileUploader"] {
  background-color: var(--ms-surface) !important;
  color: var(--ms-ink) !important;
  border-color: var(--ms-line) !important;
}
[data-baseweb="select"] * { color: var(--ms-ink); }
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: var(--ms-ink); }
div[data-testid="stRadio"] div[role="radiogroup"] label p,
div[data-testid="stCheckbox"] label p {
  color: var(--ms-body);
}
[data-testid="stAlert"] {
  background: var(--ms-surface); border: 1px solid var(--ms-line); color: var(--ms-ink);
}
[data-testid="stAlert"] p { color: var(--ms-ink); }
[data-testid="stTabs"] button { color: var(--ms-muted); }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--ms-primary); border-color: var(--ms-primary); }
[data-testid="stDataFrame"] { background: var(--ms-surface) !important; color: var(--ms-ink) !important; }
div[data-testid="stMetric"] {
  background: var(--ms-surface); border: 1px solid var(--ms-line);
  border-radius: 14px; padding: .9rem 1.05rem; box-shadow: var(--ms-shadow);
}
div[data-testid="stMetric"] label { color: var(--ms-muted); font-size: .86rem; }
div[data-testid="stExpander"] {
  background: var(--ms-surface); border: 1px solid var(--ms-line);
  border-radius: 14px; box-shadow: var(--ms-shadow);
}
div[data-testid="stExpander"] summary { font-weight: 650; }
[data-testid="stPageLink"] button {
  background: var(--ms-surface); border: 1px solid var(--ms-line);
  border-radius: 12px; font-weight: 650; color: var(--ms-primary);
}
hr { border: 0; height: 1px; background: var(--ms-line); }

/* ---------- typography helpers ---------- */
.ms-disclaimer {
  border-left: 5px solid var(--ms-rose);
  background: var(--ms-rose-soft);
  color: var(--ms-rose-ink);
  padding: .75rem .9rem; border-radius: .75rem;
  font-weight: 600; line-height: 1.5; font-size: .92rem;
}
.ms-tag {
  display: inline-block; padding: .16rem .62rem; border-radius: 999px;
  font-weight: 650; font-size: .85rem; margin-right: .4rem;
}
.ms-tag-lower   { background: var(--ms-primary-soft); color: var(--ms-primary-dark); }
.ms-tag-elevated { background: var(--ms-amber-soft); color: var(--ms-amber); }
.ms-tag-higher  { background: var(--ms-rose-soft); color: var(--ms-rose); }
.ms-tag-info    { background: var(--ms-blue-soft); color: var(--ms-blue); }
.ms-kpi { font-size: 2rem; font-weight: 750; color: var(--ms-ink); line-height: 1.15; }
.ms-kpi-label { font-size: .88rem; color: var(--ms-muted); }
.ms-note { color: var(--ms-muted); font-size: .9rem; }
.ms-quote {
  background: var(--ms-surface); border: 1px solid var(--ms-line);
  border-left: 4px solid var(--ms-primary);
  border-radius: 12px; padding: .95rem 1.1rem; color: var(--ms-body);
}
:root { --ms-radius: 16px; }

@media (max-width: 640px) {
  .stApp { font-size: 16px; }
  .ms-kpi { font-size: 1.65rem; }
  .ms-card { padding: 1rem 1.05rem; }
}
"""


def _css() -> str:
    return "<style>\n" + _root_css() + "\n" + _LAYOUT + "\n</style>"


def apply() -> None:
    """Inject the current-mode theme. Call once per page after ``set_page_config``."""
    st.markdown(_css(), unsafe_allow_html=True)


def hero(title: str, subtitle: str, *, gradient_word: str | None = None) -> None:
    """Render the calm hero banner (optional gradient-highlighted word)."""
    heading = escape(title)
    if gradient_word and gradient_word in heading:
        heading = heading.replace(
            escape(gradient_word), f'<span class="ms-grad">{escape(gradient_word)}</span>'
        )
    st.markdown(
        f'<div class="ms-hero"><h1>{heading}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def card(kicker: str, title: str, body: str, *, icon: str = "") -> str:
    """Build a static card as an HTML string (escape user-derived text!)."""
    emoji = f'<span class="ms-emoji">{icon}</span> ' if icon else ""
    return (
        '<div class="ms-card">'
        f'<span class="ms-kicker">{escape(kicker)}</span>'
        f"<h4>{emoji}{escape(title)}</h4>"
        f"<p>{body}</p>"
        "</div>"
    )


def steps(*labels: str, active: int = 0) -> None:
    """Render the progress stepper (``active`` is 0-based)."""
    cells = []
    for i, label in enumerate(labels):
        cls = "is-active" if i == active else ("is-done" if i < active else "")
        mark = "●" if i <= active else "○"
        cells.append(f'<span class="ms-step {cls}">{mark} {escape(label)}</span>')
    st.markdown(f'<div class="ms-steps">{"".join(cells)}</div>', unsafe_allow_html=True)
