"""Unit tests for the compare-and-rank helpers (``mindsense.models.ranking``)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from mindsense.models.ranking import rank_classifiers, rank_regressors


def _classification_data(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    mid = n // 2
    return x[:mid], y[:mid], x[mid:], y[mid:]


def test_rank_classifiers_sorts_by_macro_f1_and_returns_winner() -> None:
    x_train, y_train, x_val, y_val = _classification_data()
    ranking = rank_classifiers(
        {
            "logistic": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=10, random_state=42),
        },
        x_train,
        y_train,
        x_val,
        y_val,
    )
    assert ranking.best in {"logistic", "random_forest"}
    assert list(ranking.table.index) == sorted(
        ranking.table.index,
        key=lambda m: ranking.table.loc[m, "macro_f1"],
        reverse=True,
    )
    assert ranking.table["accuracy"].max() >= 0.5


def test_rank_regressors_sorts_by_rmse_ascending_multi_output() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 3))
    y = np.column_stack([2 * x[:, 0], -x[:, 1] + 1])
    mid = 50
    ranking = rank_regressors(
        {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=10, random_state=42),
        },
        x[:mid],
        y[:mid],
        x[mid:],
        y[mid:],
        target_names=["d", "a"],
    )
    assert ranking.best == "linear"
    assert list(ranking.table.index) == sorted(
        ranking.table.index, key=lambda m: ranking.table.loc[m, "rmse_mean"]
    )
    assert ranking.table["rmse_mean"].iloc[0] < ranking.table["rmse_mean"].iloc[1]
