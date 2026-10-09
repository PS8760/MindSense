"""Exp 7 — DASS-42 prognosis (multi-target regression) with model comparison.

Demographics (age/gender/education/urban/country) predict the three DASS-42
subscale scores — depression, anxiety, stress — as a multi-output regression.
Regressors are ranked on validation by mean RMSE; the winner is scored on the
held-out test split and the compare-and-rank table lands in
``reports/metrics/exp07_prognosis.json``.

sklearn-only (no xgboost) so it is safe to run in-process alongside torch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mindsense import GLOBAL_SEED
from mindsense.models.ranking import rank_regressors, slice_splits
from mindsense.utils.io import repo_path, save_json
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.prognosis")

TARGET_COLS = ["depression_score", "anxiety_score", "stress_score"]
NUMERIC_FEATURES = ["age", "education", "urban"]
CATEGORICAL_FEATURES = ["gender"]
COUNTRY_TOP = 12
QUICK_ROWS = 3000
QUICK_TREES = 50


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numerics; one-hot gender and the top countries + 'other'."""
    features = data[NUMERIC_FEATURES].copy().astype(float)
    for col in features.columns:
        features[col] = features[col].fillna(features[col].median())
    categoricals: list[pd.DataFrame] = [
        pd.get_dummies(data[CATEGORICAL_FEATURES], dtype=float)
    ]
    if "country" in data.columns:
        totals = data["country"].fillna("other").value_counts().head(COUNTRY_TOP)
        country = data["country"].fillna("other").map(
            lambda v: v if v in totals.index else "other"
        )
        categoricals.append(pd.get_dummies(country, prefix="country", dtype=float))
    return pd.concat([features, *categoricals], axis=1)


def _evaluate(best_estimator, x_test, y_test) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import mean_squared_error

    pred = np.asarray(best_estimator.predict(x_test))
    rmse_by_target = {
        name: float(np.sqrt(mean_squared_error(y_test[:, i], pred[:, i])))
        for i, name in enumerate(TARGET_COLS)
    }
    return {
        "rmse_mean": round(float(np.mean(list(rmse_by_target.values()))), 4),
        "rmse_by_target": {
            k: round(float(v), 4) for k, v in rmse_by_target.items()
        },
        "n": int(y_test.shape[0]),
    }


def train_prognosis(
    *,
    quick: bool = False,
    seed: int = GLOBAL_SEED,
    data: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and rank the DASS-42 prognosis regressors; return test metrics."""
    df = data if data is not None else pd.read_parquet(
        repo_path("data", "processed", "dass42.parquet")
    )
    df = df.dropna(subset=TARGET_COLS + ["split"]).copy()

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    if quick:
        train = train.sample(min(QUICK_ROWS, len(train)), random_state=seed)
        val = val.sample(min(QUICK_ROWS, len(val)), random_state=seed)
        test = test.sample(min(QUICK_ROWS, len(test)), random_state=seed)

    combined = pd.concat(
        [train, val, test], ignore_index=True
    ).reset_index(drop=True)
    x_all = build_features(combined).to_numpy(dtype=float)
    x_train, x_val, x_test = slice_splits(x_all, train, val, test)
    y_all = combined[TARGET_COLS].to_numpy(dtype=float)
    y_train = y_all[: len(train)]
    y_val = y_all[len(train): len(train) + len(val)]
    y_test = y_all[len(train) + len(val):]

    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor

    n_trees = QUICK_TREES if quick else 200
    estimators: dict[str, Any] = {
        "linear": LinearRegression(),
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
        ),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=seed)
        ),
    }

    ranking = rank_regressors(
        estimators, x_train, y_train, x_val, y_val,
        target_names=TARGET_COLS, seed=seed,
    )
    for name, metric in ranking.val_metrics.items():
        log.info(
            "  %-16s prognosis rmse %.3f r2 %.3f",
            name, metric["rmse_mean"], metric["r2_mean"],
        )
    log.info("exp07 best on val: %s", ranking.best)
    for i, (name, row) in enumerate(ranking.table.iterrows(), start=1):
        log.info("  rank %d %-16s rmse %.3f", i, name, row["rmse_mean"])

    test_metric = _evaluate(ranking.fitted[ranking.best], x_test, y_test)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "rows": {"train": int(y_train.shape[0]), "val": int(y_val.shape[0]), "test": int(y_test.shape[0])},
        "best_model": ranking.best,
        "ranking": ranking.table.round(4).reset_index().rename(
            columns={"index": "model"}
        ).to_dict("records"),
        "test": test_metric,
    }
    save_json(
        payload,
        (out_dir or repo_path("reports", "metrics")) / "exp07_prognosis.json",
    )
    return {
        "best": ranking.best,
        "ranking": payload["ranking"],
        "test": test_metric,
    }