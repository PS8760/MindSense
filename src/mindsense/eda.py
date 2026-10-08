"""EDA helpers for Experiment 2 (``notebooks/exp02_eda.ipynb``).

Reusable statistics and aggregation functions so the notebook stays a thin,
narrative layer (Rule 7).  Nothing here writes files except
:func:`save_plotly_html`.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Age / numeric banding
# --------------------------------------------------------------------------

AGE_BANDS: list[tuple[int, int, str]] = [
    (14, 24, "14-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 59, "45-59"),
    (60, 120, "60+"),
]


def age_band(age: float | int | None) -> str:
    """Map an age in years to a labelled band (``14-24`` … ``60+``)."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    for lo, hi, label in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return "under-14" if age < 14 else "60+"


def add_age_band(df: pd.DataFrame, col: str = "age") -> pd.DataFrame:
    """Return ``df`` with an extra ``age_band`` column (not a copy of all data)."""
    out = df.copy()
    out["age_band"] = out[col].map(age_band)
    return out


# --------------------------------------------------------------------------
# Mixed-type association measures
# --------------------------------------------------------------------------


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series (0–1)."""
    from scipy.stats import chi2_contingency  # local import: scipy is a train dep

    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    if min(kcorr - 1, rcorr - 1) <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1)))


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """Correlation ratio η (categorical → numeric), 0–1."""
    frame = pd.DataFrame({"cat": cat, "num": num}).dropna()
    if frame.empty or frame["cat"].nunique() < 2:
        return 0.0
    grand = frame["num"].mean()
    ss_total = ((frame["num"] - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = frame.groupby("cat")["num"].apply(lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    return float(np.sqrt(max(0.0, ss_between / ss_total)))


def association_matrix(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Square association matrix mixing Pearson (num×num), Cramér's V
    (cat×cat) and η (cat×num).  Absolute values, range 0–1."""
    cols = numeric + categorical
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    data = df[cols]
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if a in numeric and b in numeric:
                val = abs(data[a].corr(data[b]))
            elif a in categorical and b in categorical:
                val = cramers_v(data[a], data[b])
            else:
                cat, num = (a, b) if a in categorical else (b, a)
                val = correlation_ratio(data[cat], data[num])
            mat.loc[a, b] = mat.loc[b, a] = 0.0 if pd.isna(val) else float(val)
    return mat


def mutual_information_ranking(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    categorical: list[str] | None = None,
    seed: int = 42,
) -> pd.Series:
    """Mutual information (classification) of ``features`` vs ``target``.

    Returns a descending series of MI scores in nats.  Missing values are
    median/mode-filled on a copy purely for scoring.
    """
    from sklearn.feature_selection import mutual_info_classif

    categorical = categorical or []
    work = df[features + [target]].dropna(subset=[target]).copy()
    for col in features:
        if work[col].isna().any():
            if col in categorical:
                work[col] = work[col].fillna(work[col].mode().iloc[0])
            else:
                work[col] = work[col].fillna(work[col].median())
    cat_mask = [col in categorical for col in features]
    # Encode categoricals as codes so MI sees discrete values.
    for col, discrete in zip(features, cat_mask, strict=False):
        if discrete:
            work[col] = work[col].astype("category").cat.codes
    mi = mutual_info_classif(
        work[features], work[target], discrete_features=cat_mask, random_state=seed
    )
    return pd.Series(mi, index=features, name="mi").sort_values(ascending=False)


# --------------------------------------------------------------------------
# Text statistics
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z']+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "being", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those", "am",
    "do", "does", "did", "not", "no", "yes", "so", "as", "at", "by", "from", "about", "into",
    "over", "under", "again", "there", "here", "what", "when", "where", "who", "whom", "which",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "can", "will", "just", "don", "s", "t", "m", "o", "re", "d", "ll",
    "ve", "y"
})


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]


def top_ngrams(
    texts: pd.Series,
    labels: pd.Series,
    *,
    n: int = 2,
    top_k: int = 12,
    min_df: int = 5,
) -> dict[str, list[tuple[str, int]]]:
    """Top-k word n-grams per class (stopwords removed, ``min_df`` across
    the whole corpus).  Returns ``{class_label: [(gram, count), …]}``."""
    per_class: dict[str, Counter[str]] = {}
    global_df: Counter[str] = Counter()
    for text, label in zip(texts.fillna(""), labels, strict=False):
        toks = _tokens(text)
        grams = {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}
        if not grams:
            continue
        bucket = per_class.setdefault(str(label), Counter())
        for g in grams:
            bucket[g] += 1  # document frequency within class
            global_df[g] += 1
    kept = {g for g, c in global_df.items() if c >= min_df}
    return {
        cls: sorted(((g, c) for g, c in counter.items() if g in kept), key=lambda t: -t[1])[:top_k]
        for cls, counter in sorted(per_class.items())
    }


# A compact, self-curated emotion lexicon (≈10 stems per emotion).  This is
# an internal EDA aid — we deliberately do not bundle third-party lexicons
# (NRC etc.) whose licences we have not verified.
EMOTION_LEXICON: dict[str, tuple[str, ...]] = {
    "joy": ("happy", "joy", "glad", "love", "great", "wonderful", "enjoy", "laugh", "smile", "excited"),
    "sadness": ("sad", "cry", "alone", "empty", "hopeless", "unhappy", "miserable", "lonely", "grief", "sobbing"),
    "fear": ("afraid", "scared", "fear", "panic", "anxious", "worry", "terrified", "nervous", "dread", "phobia"),
    "anger": ("angry", "rage", "hate", "furious", "irritated", "annoyed", "frustrated", "mad", "hostile", "resent"),
    "trust": ("trust", "safe", "support", "reliable", "confident", "secure", "help", "care", "comfort", "faith"),
    "anticipation": ("hope", "expect", "plan", "prepare", "ready", "future", "goal", "look", "waiting", "soon"),
}


def emotion_rates(texts: pd.Series) -> pd.DataFrame:
    """Per-document rate of emotion words (hits / tokens) per emotion."""
    rows: list[dict[str, float]] = []
    for text in texts.fillna(""):
        toks = _tokens(str(text))
        n = max(len(toks), 1)
        rows.append(
            {
                emo: sum(any(t.startswith(stem) for stem in stems) for t in toks) / n
                for emo, stems in EMOTION_LEXICON.items()
            }
        )
    return pd.DataFrame(rows, index=texts.index)


def emotion_trends(texts: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Mean emotion-word rate per class (rows = class, columns = emotion)."""
    rates = emotion_rates(texts)
    rates["label"] = labels.values
    return rates.groupby("label").mean()


def vader_compound(texts: pd.Series, *, limit: int | None = None) -> pd.Series:
    """Mean VADER compound sentiment per text (optionally truncated for speed)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    series = texts.fillna("").astype(str)
    if limit is not None and len(series) > limit:
        series = series.iloc[:limit]
    return series.map(lambda t: analyzer.polarity_scores(t)["compound"])


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------


def save_plotly_html(fig: Any, name: str) -> Path:
    """Write a plotly figure to ``reports/figures/<name>.html`` (interactive
    artefacts for the Population Insights page and the report)."""
    from mindsense.utils.io import ensure_dir, repo_path

    out = repo_path("reports", "figures", f"{name}.html")
    ensure_dir(out.parent)
    fig.write_html(out, include_plotlyjs="cdn")
    return out
