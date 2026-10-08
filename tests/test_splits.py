"""Split-construction and leakage-guard invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mindsense.data.features import leakage_report, stratified_splits, text_overlap


def _risk_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "mh_risk": rng.integers(0, 2, n),
            "population": rng.choice(["student", "professional"], n),
        }
    )


def test_splits_are_disjoint_and_complete():
    df = _risk_frame()
    splits = stratified_splits(df)
    parts = [set(v) for v in splits.values()]
    assert not (parts[0] & parts[1]) and not (parts[1] & parts[2])
    assert parts[0] | parts[1] | parts[2] == set(range(len(df)))


def test_split_ratios_within_tolerance():
    df = _risk_frame(1000)
    splits = stratified_splits(df)
    n = len(df)
    assert abs(len(splits["train"]) / n - 0.70) < 0.03
    assert abs(len(splits["val"]) / n - 0.15) < 0.03
    assert abs(len(splits["test"]) / n - 0.15) < 0.03


def test_split_is_deterministic_for_seed():
    df = _risk_frame(200)
    assert stratified_splits(df) == stratified_splits(df)


def test_every_split_contains_every_stratum():
    df = _risk_frame(400)
    strata = df.astype(str).agg("|".join, axis=1)
    splits = stratified_splits(df)
    for name, idx in splits.items():
        present = set(strata.iloc[idx])
        assert present == set(strata), f"{name} is missing strata"


def test_stratified_class_rate_stable_across_splits():
    df = _risk_frame(600)
    splits = stratified_splits(df)
    rates = [df["mh_risk"].iloc[idx].mean() for idx in splits.values()]
    assert max(rates) - min(rates) < 0.05


def test_bad_ratios_raise():
    with pytest.raises(AssertionError):
        stratified_splits(_risk_frame(50), train=0.5, val=0.5, test=0.5)


def test_text_overlap_counts_intersection():
    a = ["alpha", "beta", "gamma"]
    b = ["beta", "gamma", "delta"]
    out = text_overlap(a, b)
    assert out == {"n_a": 3, "n_b": 3, "overlap": 2, "examples": ["beta", "gamma"]}


def test_leakage_report_flags_cross_corpus_duplicates():
    sentiment = pd.DataFrame({"text_norm": ["hello world", "unique a", "shared"]})
    dreaddit = pd.DataFrame({"text_norm": ["shared", "hello world", "other"]})
    report = leakage_report(sentiment, dreaddit)
    overlap = report["sentiment_vs_dreaddit"]
    assert overlap["overlap"] == 2
    assert set(overlap["examples"]) == {"hello world", "shared"}
