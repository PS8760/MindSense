"""App smoke: render Home + every page with Streamlit's AppTest.

Used by ``make smoke`` and as the CI fallback
(``python -c "import app.home_smoke" || python scripts/smoke_app.py``).
Exits non-zero if any page raises.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    REPO / "app" / "Home.py",
    *sorted((REPO / "app" / "pages").glob("*.py")),
]


def run_page(path: Path) -> str | None:
    """Run one page; return None on success, a failure description otherwise."""
    try:
        at = AppTest.from_file(str(path), default_timeout=60)
        at.run()
    except Exception:  # noqa: BLE001 - smoke reports everything
        return f"{path.name}: crashed at run()\n{traceback.format_exc()}"
    if at.exception:
        return f"{path.name}: exception\n{at.exception!r}"
    return None


def main() -> int:
    failures: list[str] = []
    for page in PAGES:
        failure = run_page(page)
        if failure:
            failures.append(failure)
            print(f"FAIL {page.name}")
        else:
            print(f"ok   {page.name}")
    if failures:
        print("\n--- failures ---")
        for failure in failures:
            print(failure)
        return 1
    print(f"\nAll {len(PAGES)} pages rendered cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
