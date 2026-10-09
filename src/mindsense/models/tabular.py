"""Exp 4 — tabular risk screen (student + professional) with model comparison.

Feeds ``data/processed/risk_harmonized.parquet`` (splits carved by the data
pipeline) into several sklearn classifiers and ranks them on validation by
macro-F1/accuracy. The winner is scored on the held-out test split and the
whole compare-and-rank table lands in ``reports/metrics/exp04_tabular.json``.

sklearn-only (no xgboost) so it is safe to run in-process alongside torch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mindsense import GLOBAL_SEED
from mindsense.models.ranking import rank_classifiers, slice_splits
from mindsense.utils.io import repo_path, save_json
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.tabular")

NUMERIC_FEATURES = [
    "age",
    "sleep_hours",
    "stress_pressure_score",
    "satisfaction_score",
    "work_study_hours",
    "financial_stress",
    "diet_quality",
]
CATEGORICAL_FEATURES = ["gender", "population", "family_history", "suicidal_thoughts"]
TARGET = "mh_risk"

QUICK_ROWS = 3000
QUICK_ESTIMATORS = 50


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns and one-hot the categoricals."""
    features = data[NUMERIC_FEATURES].copy()
    for col in features.columns:
        features[col] = features[col].fillna(features[col].median())
    categoricals = pd.get_dummies(
        data[CATEGORICAL_FEATURES].astype("category"), dtype=float
    )
    return pd.concat([features, categoricals], axis=1)


def train_tabular(
    *,
    quick: bool = False,
    seed: int = GLOBAL_SEED,
    data: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and rank the tabular risk-screen models; return test metrics."""
    df = data if data is not None else pd.read_parquet(
        repo_path("data", "processed", "risk_harmonized.parquet")
    )
    df = df.dropna(subset=[TARGET]).copy()

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
    y_all = combined[TARGET].astype(int).to_numpy()
    y_train = y_all[: len(train)]
    y_val = y_all[len(train): len(train) + len(val)]
    y_test = y_all[len(train) + len(val):]

    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    n_trees = QUICK_ESTIMATORS if quick else 300
    estimators: dict[str, Any] = {
        "logistic": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=n_trees, n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }

    ranking = rank_classifiers(estimators, x_train, y_train, x_val, y_val, seed=seed)
    for name, metric in ranking.val_metrics.items():
        log.info(
            "  %-16s tabular val acc %.3f macro-F1 %.3f",
            name, metric["accuracy"], metric["macro_f1"],
        )
    log.info("exp04 best on val: %s", ranking.best)
    for i, (name, row) in enumerate(ranking.table.iterrows(), start=1):
        log.info("  rank %d %-16s macro-F1 %.3f", i, name, row["macro_f1"])

    best_estimator = ranking.fitted[ranking.best]
    test_metric = _metrics_for(best_estimator, x_test, y_test)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "rows": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "best_model": ranking.best,
        "ranking": ranking.table.round(4).reset_index().rename(
            columns={"index": "model"}
        ).to_dict("records"),
        "test": test_metric,
    }
    save_json(
        payload,
        (out_dir or repo_path("reports", "metrics")) / "exp04_tabular.json",
    )
    return{
        "best": ranking.best,
        "ranking": payload["ranking"],
        "test": test_metric,
    }


def _metrics_for(best_estimator, x_test, y_test) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    test_pred = best_estimator.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_test, test_pred, average="macro", zero_division=0)), 4
        ),
        "n": int(len(y_test)),
    }