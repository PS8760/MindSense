"""Shared pytest fixtures: repo root importable (for ``import app...``)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def _disable_groq_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite offline: inference must never call the Groq API."""
    monkeypatch.setenv("MINDSENSE_DISABLE_GROQ", "1")
