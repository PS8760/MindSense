"""Split construction and leakage guards (Exp 1 outputs, reused by Exp 2+).

Splits are stratified on the requested strata (``mh_risk x population`` for
the risk table, ``label`` for sentiment, ``depression_band`` for DASS) with
the global seed, and written into the ``split`` column of the processed
parquet (single source of truth — see ``docs/DATA_DICTIONARY.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mindsense import GLOBAL_SEED
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.features")


def stratified_splits(
    df: pd.DataFrame,
    *,
    strata_cols: tuple[str, ...] = ("mh_risk", "population"),
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = GLOBAL_SEED,
) -> dict[str, list[int]]:
    """Return {train/val/test: positional indices} stratified on strata_cols."""
    assert abs(train + val + test - 1.0) < 1e-9, "split ratios must sum to 1"
    strata = df[list(strata_cols)].astype(str).agg("|".join, axis=1)
    idx = np.arange(len(df))
    rest_frac = val + test
    train_idx, rest_idx = train_test_split(
        idx, train_size=train, random_state=seed, stratify=strata
    )
    rel_val = val / rest_frac
    val_rel_idx, test_idx = train_test_split(
        rest_idx, train_size=rel_val, random_state=seed,
        stratify=strata[rest_idx],
    )
    splits = {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_rel_idx),
        "test": sorted(int(i) for i in test_idx),
    }
    for name, rows in splits.items():
        log.info("split %-5s: %d rows", name, len(rows))
    return splits


def text_overlap(a: pd.Series | list[str], b: pd.Series | list[str]) -> dict[str, Any]:
    """Exact-overlap count between two normalised text collections."""
    sa, sb = set(a), set(b)
    inter = sa & sb
    return {
        "n_a": len(sa),
        "n_b": len(sb),
        "overlap": len(inter),
        "examples": sorted(inter)[:5],
    }


def leakage_report(sentiment: pd.DataFrame, dreaddit: pd.DataFrame) -> dict[str, Any]:
    """Cross-corpus duplicate check between the two volunteer text datasets."""
    report = {
        "sentiment_vs_dreaddit": text_overlap(sentiment["text_norm"], dreaddit["text_norm"]),
    }
    log.info("text overlap sentiment/dreaddit: %d", report["sentiment_vs_dreaddit"]["overlap"])
    return report
