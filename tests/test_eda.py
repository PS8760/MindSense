"""Unit tests for the EDA helper library (``mindsense.eda``).

All inputs are hand-built so the suite runs on a clean CI clone with no
processed data present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mindsense import eda

# --------------------------------------------------------------------------- #
# age bands
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("age", "band"),
    [
        (14, "14-24"),
        (24, "14-24"),
        (25, "25-34"),
        (34, "25-34"),
        (35, "35-44"),
        (45, "45-59"),
        (60, "60+"),
        (95, "60+"),
        (13, "under-14"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_age_band_edges(age, band: str) -> None:
    assert eda.age_band(age) == band


def test_add_age_band_adds_column_and_keeps_index() -> None:
    df = pd.DataFrame({"age": [20, 30, np.nan]}, index=[7, 8, 9])
    out = eda.add_age_band(df)
    assert list(out["age_band"]) == ["14-24", "25-34", "unknown"]
    assert list(out.index) == [7, 8, 9]
    assert "age_band" not in df.columns  # input untouched


# --------------------------------------------------------------------------- #
# association measures
# --------------------------------------------------------------------------- #

def test_cramers_v_perfect_dependence_and_independence() -> None:
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series([0, 0, 1, 1])
    assert eda.cramers_v(x, y) == pytest.approx(1.0)
    x_ind = pd.Series(["a", "b", "a", "b"])
    assert eda.cramers_v(x_ind, y) == pytest.approx(0.0)


def test_cramers_v_degenerate_table_is_zero() -> None:
    x = pd.Series(["a", "a", "a"])
    y = pd.Series([0, 1, 0])
    assert eda.cramers_v(x, y) == 0.0


def test_correlation_ratio_bounds() -> None:
    cat = pd.Series(["a", "a", "b", "b"])
    num = pd.Series([1.0, 1.0, 2.0, 2.0])  # category explains everything
    assert eda.correlation_ratio(cat, num) == pytest.approx(1.0)
    assert eda.correlation_ratio(cat, pd.Series([5.0] * 4)) == 0.0  # constant numeric
    assert eda.correlation_ratio(pd.Series(["a"] * 4), num) == 0.0  # one category


def test_association_matrix_shape_symmetry_and_range() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num_a": rng.normal(size=40),
            "num_b": rng.normal(size=40),
            "cat_a": rng.choice(["x", "y"], 40),
            "cat_b": rng.choice(["p", "q", "r"], 40),
        }
    )
    mat = eda.association_matrix(df, numeric=["num_a", "num_b"], categorical=["cat_a", "cat_b"])
    assert mat.shape == (4, 4)
    assert (mat.to_numpy() == mat.to_numpy().T).all()
    assert np.allclose(np.diag(mat), 1.0)
    assert ((mat >= 0) & (mat <= 1)).to_numpy().all()
    assert mat.loc["num_a", "num_b"] == pytest.approx(abs(df["num_a"].corr(df["num_b"])))


def test_mutual_information_ranking_orders_informative_feature_first() -> None:
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    df = pd.DataFrame(
        {"signal": signal, "noise": rng.normal(size=n), "y": (signal > 0).astype(int)}
    )
    ranking = eda.mutual_information_ranking(df, "y", ["signal", "noise"], seed=42)
    assert list(ranking.index)[:1] == ["signal"]
    assert ranking["signal"] > 0.1
    assert ranking["signal"] > ranking["noise"]
    assert ranking.is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

def test_top_ngrams_per_class_stopwords_filtered() -> None:
    texts = pd.Series(["hopeless empty cry"] * 6 + ["happy joyful love the on"] * 6)
    labels = pd.Series(["low"] * 6 + ["high"] * 6)
    grams = eda.top_ngrams(texts, labels, n=2, top_k=5, min_df=1)
    assert set(grams) == {"high", "low"}
    assert grams["low"][0] in {("hopeless empty", 6), ("empty cry", 6)}
    assert grams["high"][0][0] in {"happy joyful", "joyful love"}
    assert all("the" not in g for lst in grams.values() for g, _ in lst)


def test_emotion_lexicon_shape() -> None:
    assert set(eda.EMOTION_LEXICON) == {
        "joy", "sadness", "fear", "anger", "trust", "anticipation",
    }
    assert all(len(stems) >= 8 for stems in eda.EMOTION_LEXICON.values())


def test_emotion_rates_and_trends_on_known_texts() -> None:
    texts = pd.Series(
        ["I am happy joyful love", "I feel scared afraid panic"],
        index=[10, 11],
    )
    rates = eda.emotion_rates(texts)
    assert list(rates.index) == [10, 11]
    assert rates.loc[10, "joy"] > rates.loc[10, "fear"]
    assert rates.loc[11, "fear"] > rates.loc[11, "joy"]
    assert ((rates >= 0) & (rates <= 1)).to_numpy().all()

    labels = pd.Series(["joyful", "scared"], index=[10, 11])
    trends = eda.emotion_trends(texts, labels)
    assert set(trends.index) == {"joyful", "scared"}
    assert trends.loc["joyful", "joy"] > trends.loc["joyful", "fear"]
    assert trends.loc["scared", "fear"] > trends.loc["scared", "joy"]


def test_vader_compound_orders_positive_above_negative() -> None:
    texts = pd.Series(["I love this wonderful day", "I hate this terrible day"])
    scores = eda.vader_compound(texts)
    assert scores.iloc[0] > scores.iloc[1]
    assert scores.between(-1, 1).all()


def test_save_plotly_html_writes_file(tmp_path, monkeypatch) -> None:
    import plotly.graph_objects as go

    from mindsense.utils import io as mindsense_io

    monkeypatch.setattr(
        mindsense_io, "repo_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    fig = go.Figure(data=go.Bar(x=[1, 2], y=[3, 4]))
    out = eda.save_plotly_html(fig, "unit_test_fig")
    assert out.exists()
    assert out.read_text(encoding="utf-8")[:15].lower().startswith("<!doctype html")
