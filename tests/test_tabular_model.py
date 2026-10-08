"""Unit tests for the Exp 4 tabular risk screen (``mindsense.models.tabular``).

Runs on a tiny synthetic frame (no network, no data/processed dependency) so
the suite stays fast and CI-friendly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from mindsense.models.tabular import build_features, train_tabular

NUMERIC = [
    "age",
    "sleep_hours",
    "stress_pressure_score",
    "satisfaction_score",
    "work_study_hours",
    "financial_stress",
    "diet_quality",
]
CATEGORICAL = ["gender", "population", "family_history", "suicidal_thoughts"]


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = int(n / 3)
    return pd.DataFrame(
        {
            **{col: rng.normal(5, 2, n) for col in NUMERIC},
            "gender": rng.choice(["male", "female", "other"], n),
            "population": rng.choice(["student", "professional"], n),
            "family_history": rng.integers(0, 2, n),
            "suicidal_thoughts": rng.integers(0, 2, n),
            "mh_risk": rng.integers(0, 2, n),
            "split": ["train"] * rows + ["val"] * rows + ["test"] * rows,
        }
    )


def test_build_features_imputes_and_one_hots() -> None:
    df = _frame(60)
    df.loc[df.index[0], "age"] = np.nan
    x = build_features(df)
    assert not x.isna().any().any()
    assert x.shape[0] == 60
    assert any(c.startswith("gender_") for c in x.columns)


def test_train_tabular_ranks_models_and_reports_test_metrics() -> None:
    result = train_tabular(quick=True, data=_frame(90))
    assert result["best"] in {"logistic", "random_forest", "gradient_boosting"}
    assert result["test"]["n"] == len(_frame(90)[_frame(90)["split"] == "test"])
    assert len(result["ranking"]) == 3
    ordered = [row["model"] for row in result["ranking"]]
    assert ordered == sorted(
        ordered,
        key=lambda m: next(r["macro_f1"] for r in result["ranking"] if r["model"] == m),
        reverse=True,
    )
