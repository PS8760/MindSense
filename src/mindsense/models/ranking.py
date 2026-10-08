"""Torch-free compare-and-rank helpers for the sklearn trainers (exp04/06/07).

Every experiment fits several model types on the shared train split, scores
them on validation, and returns a ranked leaderboard plus the winner — not
just the best model's numbers. sklearn-only: safe to fit in-process even
after torch has been imported (see docs/DECISIONS.md row 25), unlike xgboost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mindsense import GLOBAL_SEED
from mindsense.models.metrics import classification_metrics


def _int_labels(y: np.ndarray) -> list[int]:
    return sorted({int(v) for v in np.asarray(y)})


def slice_splits(
    x_all: np.ndarray, *splits: pd.DataFrame
) -> tuple[np.ndarray, ...]:
    """Slice rows back out of a feature matrix built on concatenated splits.

    Ensures every split shares the same feature universe (one-hot columns
    from all splits), so ``predict`` never sees a mismatch.
    """
    out: list[np.ndarray] = []
    start = 0
    for part in splits:
        end = start + len(part)
        out.append(x_all[start:end])
        start = end
    return tuple(out)


def encode_labels(
    *arrays: np.ndarray,
) -> tuple[list[np.ndarray], list[str]]:
    """Integer-encode labels (numeric or string) across all provided arrays.

    Returns ``(encoded_arrays, class_names)`` where ``class_names[i]`` names
    the integer ``i`` — the convention ``classification_metrics`` expects.
    """
    combined = np.concatenate([np.asarray(a).ravel() for a in arrays])
    numeric = np.issubdtype(combined.dtype, np.number) or np.issubdtype(
        combined.dtype, np.bool_
    )
    keys = sorted({int(v) for v in combined}) if numeric else sorted(
        {str(v) for v in combined}
    )
    lookup = {key: i for i, key in enumerate(keys)}
    encoded = [
        np.array(
            [lookup[int(v)] if numeric else lookup[str(v)] for v in np.asarray(a)],
            dtype=np.int64,
        )
        for a in arrays
    ]
    return encoded, [str(k) for k in keys]


class ClassifierRanking:
    """Result of ranking several classifiers on a validation split."""

    def __init__(
        self,
        table: pd.DataFrame,
        best: str,
        fitted: dict[str, Any],
        val_metrics: dict[str, dict[str, Any]],
    ) -> None:
        self.table = table
        self.best = best
        self.fitted = fitted
        self.val_metrics = val_metrics


def rank_classifiers(
    estimators: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = GLOBAL_SEED,
) -> ClassifierRanking:
    """Fit every estimator on train, rank by macro-F1/accuracy on validation."""
    del seed  # estimators are pre-seeded by the caller
    classes = [str(c) for c in _int_labels(np.concatenate([y_train, y_val]))]
    rows: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    val_metrics: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        estimator.fit(x_train, np.asarray(y_train))
        fitted[name] = estimator
        metric = classification_metrics(
            np.asarray(y_val), estimator.predict(x_val), classes
        )
        rows[name] = {
            "accuracy": metric["accuracy"],
            "macro_f1": metric["macro_f1"],
            "weighted_f1": metric["weighted_f1"],
        }
        val_metrics[name] = metric
    table = pd.DataFrame(rows).T.sort_values(["macro_f1", "accuracy"], ascending=False)
    return ClassifierRanking(table, str(table.index[0]), fitted, val_metrics)


class RegressorRanking:
    """Result of ranking several (possibly multi-output) regressors."""

    def __init__(
        self,
        table: pd.DataFrame,
        best: str,
        fitted: dict[str, Any],
        val_metrics: dict[str, dict[str, Any]],
    ) -> None:
        self.table = table
        self.best = best
        self.fitted = fitted
        self.val_metrics = val_metrics


def rank_regressors(
    estimators: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    target_names: list[str] | None = None,
    seed: int = GLOBAL_SEED,
) -> RegressorRanking:
    """Fit every estimator on train, rank by RMSE (ascending) on validation.

    Works for single-output and multi-output regressors alike; multi-output
    error is averaged over ``target_names`` (default: one per target column).
    """
    del seed
    from sklearn.metrics import mean_squared_error, r2_score

    y_val = np.asarray(y_val)
    y_train = np.asarray(y_train)
    n_targets = 1 if y_val.ndim == 1 else y_val.shape[1]
    names = target_names or [f"target_{i}" for i in range(n_targets)]
    rows: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    val_metrics: dict[str, dict[str, Any]] = {}
    for model_name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        fitted[model_name] = estimator
        pred = estimator.predict(x_val)
        pred = np.asarray(pred)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        rmse_by_target = {
            name: float(np.sqrt(mean_squared_error(y_val[:, i], pred[:, i])))
            for i, name in enumerate(names)
        }
        r2s = [float(r2_score(y_val[:, i], pred[:, i])) for i in range(n_targets)]
        rows[model_name] = {
            "rmse_mean": float(np.mean(list(rmse_by_target.values()))),
            "r2_mean": float(np.mean(r2s)),
            **{f"rmse_{name}": v for name, v in rmse_by_target.items()},
        }
        val_metrics[model_name] = {
            "rmse_mean": rows[model_name]["rmse_mean"],
            "r2_mean": rows[model_name]["r2_mean"],
            "rmse_by_target": rmse_by_target,
        }
    table = pd.DataFrame(rows).T.sort_values("rmse_mean", ascending=True)
    return RegressorRanking(table, str(table.index[0]), fitted, val_metrics)