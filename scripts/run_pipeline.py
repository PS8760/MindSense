#!/usr/bin/env python3
"""Run the Experiment 1 data pipeline (Milestone 2).

Thin CLI over :func:`mindsense.data.pipeline.run_pipeline`::

    python scripts/run_pipeline.py

Produces the processed parquets, stratified splits, quality/leakage reports
and the generated data dictionary under ``docs/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindsense.data.pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    run_pipeline()
    sys.exit(0)
