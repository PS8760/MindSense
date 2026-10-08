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
