"""Exp 6 — mental-health text screening (sentiment_mh corpus) with model comparison.

TF-IDF vectors over the cleaned corpus feed several classifiers which are
ranked on validation by macro-F1/accuracy (7-class: Anxiety, Depression,
Stress, Suicidal, Bipolar, Personality disorder, Normal). The winner is scored
on the held-out test split and the compare-and-rank table lands in
``reports/metrics/exp06_text.json``.

sklearn-only (no xgboost) so it is safe to run in-process alongside torch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mindsense import GLOBAL_SEED
from mindsense.models.ranking import rank_classifiers
from mindsense.utils.io import repo_path, save_json
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.text")

LABEL = "label"
TEXT_COL = "text_norm"
QUICK_ROWS = 3000
QUICK_MAX_FEATURES = 10000


def train_text(
    *, quick: bool = False, seed: int = GLOBAL_SEED, data: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Train and rank the text-screening models; return test metrics."""
    df = data if data is not None else pd.read_parquet(
        repo_path("data", "processed", "sentiment_mh.parquet")
    )
    text_col = TEXT_COL if TEXT_COL in df.columns else "text"
    df = df[[text_col, LABEL, "split"]].dropna().copy()

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    if quick:
        train = train.sample(min(QUICK_ROWS, len(train)), random_state=seed)
        val = val.sample(min(QUICK_ROWS, len(val)), random_state=seed)
        test = test.sample(min(QUICK_ROWS, len(test)), random_state=seed)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        max_features=QUICK_MAX_FEATURES if quick else 60000,
        min_df=2,
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(train[text_col])
    x_val = vectorizer.transform(val[text_col])
    x_test = vectorizer.transform(test[text_col])
    y_train, y_val, y_test = (
        train[LABEL].astype(str).to_numpy(),
        val[LABEL].astype(str).to_numpy(),
        test[LABEL].astype(str).to_numpy(),
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    estimators: dict[str, Any] = {
        "naive_bayes": MultinomialNB(alpha=1.0),
        "logistic": LogisticRegression(max_iter=1000, random_state=seed),
        "linear_svc": LinearSVC(max_iter=3000, random_state=seed),
    }

    from mindsense.models.ranking import encode_labels

    (y_train_int, y_val_int, y_test_int), classes = encode_labels(
        y_train, y_val, y_test
    )

    ranking = rank_classifiers(
        estimators, x_train, y_train_int, x_val, y_val_int, seed=seed
    )
    for name, metric in ranking.val_metrics.items():
        log.info(
            "  %-16s text acc %.3f macro-F1 %.3f",
            name, metric["accuracy"], metric["macro_f1"],
        )
    log.info("exp06 best on val: %s", ranking.best)
    for i, (name, row) in enumerate(ranking.table.iterrows(), start=1):
        log.info("  rank %d %-16s macro-F1 %.3f", i, name, row["macro_f1"])

    best_estimator = ranking.fitted[ranking.best]
    from mindsense.models.metrics import classification_metrics

    test_metric = classification_metrics(
        y_test_int, best_estimator.predict(x_test), classes
    )

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "text_column": text_col,
        "vocabulary": int(x_train.shape[1]),
        "rows": {"train": int(y_train.size), "val": int(y_val.size), "test": int(y_test.size)},
        "best_model": ranking.best,
        "ranking": ranking.table.round(4).reset_index().rename(
            columns={"index": "model"}
        ).to_dict("records"),
        "test": {k: test_metric[k] for k in ("accuracy", "macro_f1", "weighted_f1", "n")},
    }
    save_json(payload, repo_path("reports", "metrics", "exp06_text.json"))
    return {
        "best": ranking.best,
        "ranking": payload["ranking"],
        "test": payload["test"],
    }