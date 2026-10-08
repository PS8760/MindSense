"""Calm, accessible visual theme for MindSense (Section 8, global UI).

Soft blues/greens, large readable type, mobile-friendly paddings, light/dark
friendly (colours are tuned for light mode; dark mode falls back to the
Streamlit defaults because our accent colours pass contrast on both).
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
  --ms-blue: #3B7DD8;
  --ms-green: #2FA88B;
  --ms-sand: #E8A13A;
  --ms-red: #C0504D;
  --ms-ink: #1F2A37;
}
.stApp { font-size: 17px; }
h1, h2, h3 { letter-spacing: -0.01em; }
h1 { color: var(--ms-blue); }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #F3F8FD 0%, #F1FBF7 100%);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--ms-ink); }
.ms-disclaimer {
  border-left: 5px solid var(--ms-red);
  background: #FDF3F2;
  color: #7A2420;
  padding: 0.75rem 0.9rem;
  border-radius: 0.5rem;
  font-weight: 600;
  line-height: 1.45;
}
.ms-tag {
  display: inline-block;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.85rem;
  margin-right: 0.4rem;
}
.ms-tag-lower   { background: #E3F5EF; color: #14664F; }
.ms-tag-elevated { background: #FDF3E3; color: #8A5A11; }
.ms-tag-higher  { background: #FBE9E8; color: #8C2F2B; }
.ms-tag-info    { background: #E8F1FC; color: #215AA8; }
.ms-kpi { font-size: 2.1rem; font-weight: 700; color: var(--ms-blue); line-height: 1.1; }
.ms-kpi-label { font-size: 0.9rem; color: #5B6B7C; }
div[data-testid="stExpander"] summary { font-weight: 600; }
@media (max-width: 640px) {
  .stApp { font-size: 16px; }
  .ms-kpi { font-size: 1.7rem; }
}
</style>
"""


def apply() -> None:
    """Inject the theme. Call once per page, right after ``set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)
