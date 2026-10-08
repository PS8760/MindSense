"""App smoke tests: every page must render without exceptions (Section 12)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
PAGE_PATHS = [REPO / "app" / "Home.py", *sorted((REPO / "app" / "pages").glob("*.py"))]


def test_all_pages_exist():
    assert len(PAGE_PATHS) >= 8, "Home + 7 pages expected"


@pytest.mark.parametrize("page", PAGE_PATHS, ids=lambda p: p.name)
def test_page_renders(page: Path):
    at = AppTest.from_file(str(page), default_timeout=60)
    at.run()
    assert not at.exception, f"{page.name} raised {at.exception!r}"


def test_inference_returns_available_flag_not_raise():
    from mindsense import inference

    result = inference.predict_risk({"population": "student", "age": 22, "gender": "male"})
    assert result["available"] is False  # models not trained yet
    assert result["message"]
    status = inference.availability()
    assert status["risk"]["available"] is False


def test_check_in_submit_shows_results_offline():
    from app.components import ai

    original = ai._cached_suggestions

    def _offline(*_args, **_kwargs):
        raise RuntimeError("offline for test")

    ai._cached_suggestions = _offline
    try:
        at = AppTest.from_file(str(REPO / "app" / "pages" / "1_Check_In.py"), default_timeout=90)
        at.run()
        assert not at.exception
        buttons = at.button
        submit = [b for b in buttons if b.label == "See my results"]
        assert submit, "submit button should exist"
        submit[0].click()
        at.run()
        assert not at.exception, f"check-in flow raised {at.exception!r}"
        assert any(s.value == "What stands out" for s in at.subheader)
        assert "Offline suggestions" in " ".join(c.value for c in at.caption)
    finally:
        ai._cached_suggestions = original
