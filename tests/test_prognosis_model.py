"""Unit tests for the Exp 7 DASS-42 prognosis (``mindsense.models.prognosis``).

Small synthetic frame; no network / data/processed dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from mindsense.models.prognosis import TARGET_COLS, build_features, train_prognosis


def _frame(n: int) -> pd.DataFrame:
    rows = n // 3
    rng = np.random.default_rng(7)
    country = rng.choice(["US", "IN", "GB", "PL", "NG", "AU"], n)
    return pd.DataFrame(
        {
            "age": rng.integers(16, 60, n),
            "education": rng.integers(1, 6, n),
            "urban": rng.integers(1, 4, n),
            "gender": rng.choice(["female", "male", "other"], n),
            "country": country,
            **{col: rng.normal(12, 6, n).clip(0, 42) for col in TARGET_COLS},
            "split": ["train"] * rows + ["val"] * rows + ["test"] * rows,
        }
    )


def test_build_features_encodes_demographics() -> None:
    x = build_features(_frame(30))
    assert x.shape[0] == 30
    assert any(c.startswith("gender_") for c in x.columns)
    assert any(c.startswith("country_") for c in x.columns)


def test_train_prognosis_ranks_regressors_and_reports_rmse() -> None:
    result = train_prognosis(quick=True, data=_frame(90))
    assert result["best"] in {"linear", "random_forest", "gradient_boosting"}
    assert result["test"]["rmse_mean"] >= 0.0
    assert set(TARGET_COLS) <= set(result["test"]["rmse_by_target"])
    assert len(result["ranking"]) == 3
