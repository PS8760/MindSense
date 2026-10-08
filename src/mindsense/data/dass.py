"""DASS-42 scoring, validity screening and severity bands (Exp 1 / Exp 4).

Sources
-------
* Item key and 0–3 recoding: Open Psychometrics codebook shipped in the raw zip
  (responses are recorded 1–4 and recoded to 0–3).
* Subscale item membership and severity cut-offs: Lovibond & Lovibond (1995),
  DASS manual, via the UNSW DASS site (http://www2.psy.unsw.edu.au/dass/).

Depression items: 3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42
Stress items:     1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39
Anxiety items:    2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41

Severity bands apply to the **raw 14-item sum (0–42)** — the "multiply by 2"
step in DASS-21 exists only to make DASS-21 comparable to DASS-42 and must
NOT be applied here (checked against DASS-21 and DASS-42 cut-off tables,
which align once doubled: 21-item moderate 7–10 → 14–20, etc.).

    Depression: 0–9 normal, 10–13 mild, 14–20 moderate, 21–27 severe, 28+ extremely severe
    Anxiety:    0–7 normal,  8–9 mild, 10–14 moderate, 15–19 severe, 20+ extremely severe
    Stress:     0–14 normal, 15–18 mild, 19–25 moderate, 26–33 severe, 34+ extremely severe
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mindsense.utils.io import repo_path
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.dass")

DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
STRESS_ITEMS = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
ANXIETY_ITEMS = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]

SUBSCALE_ITEMS = {
    "depression": DEPRESSION_ITEMS,
    "anxiety": ANXIETY_ITEMS,
    "stress": STRESS_ITEMS,
}

SEVERITY_BANDS: dict[str, list[tuple[str, int]]] = {
    "depression": [("Normal", 9), ("Mild", 13), ("Moderate", 20), ("Severe", 27),
                   ("Extremely severe", 42)],
    "anxiety": [("Normal", 7), ("Mild", 9), ("Moderate", 14), ("Severe", 19),
                ("Extremely severe", 42)],
    "stress": [("Normal", 14), ("Mild", 18), ("Moderate", 25), ("Severe", 33),
               ("Extremely severe", 42)],
}

BAND_ORDER = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]
FAKE_VOCAB_ITEMS = ["VCL6", "VCL9", "VCL12"]
MIN_TEST_SECONDS = 60  # 42 items cannot credibly be answered faster than this

# Worked examples used by the unit tests (score, expected band).
WORKED_EXAMPLES = {
    "depression": [(9, "Normal"), (12, "Mild"), (17, "Moderate"), (24, "Severe"),
                   (30, "Extremely severe")],
    "anxiety": [(7, "Normal"), (9, "Mild"), (12, "Moderate"), (17, "Severe"),
                (22, "Extremely severe")],
    "stress": [(14, "Normal"), (16, "Mild"), (22, "Moderate"), (30, "Severe"),
               (36, "Extremely severe")],
}


def raw_dass_path() -> Path:
    return repo_path("data", "raw", "dass42", "DASS_data_21.02.19", "data.csv")


def severity_band(subscale: str, score: int) -> str:
    """Map a subscale total (0–42) onto its severity band."""
    for band, upper in SEVERITY_BANDS[subscale]:
        if score <= upper:
            return band
    return "Extremely severe"  # pragma: no cover - defensive


def load_dass() -> pd.DataFrame:
    """Load the raw DASS-42 responses (1–4 item scale, pre-recoding)."""
    return pd.read_csv(raw_dass_path(), sep="\t")


def recode_items(df: pd.DataFrame) -> pd.DataFrame:
    """Recode ``QnA`` responses from 1–4 to 0–3 (in place copy)."""
    out = df.copy()
    for i in range(1, 43):
        col = f"Q{i}A"
        out[col] = pd.to_numeric(out[col], errors="coerce") - 1
    return out


def validity_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that pass the validity screen.

    Excluded (per Section 5.5): respondents endorsing ≥ 2 of the three fake
    vocabulary words, impossible ages, straight-liners (all 42 items equal)
    and implausibly fast completions (< 60 s test time). The ≥ 2 fake-word
    threshold (rather than ≥ 1) is documented in ``docs/DECISIONS.md``.
    """
    fake = df[FAKE_VOCAB_ITEMS].sum(axis=1)
    item_cols = [f"Q{i}A" for i in range(1, 43)]
    nunique = df[item_cols].nunique(axis=1)
    age = pd.to_numeric(df["age"], errors="coerce")
    test_time = pd.to_numeric(df.get("testelapse"), errors="coerce")
    ok = (fake < 2) & age.between(10, 100) & (nunique > 1)
    if test_time is not None:
        ok &= test_time.isna() | (test_time >= MIN_TEST_SECONDS)
    return ok.fillna(False)


def score_subscale(row: pd.Series, subscale: str) -> int:
    """Sum the 14 items of one subscale for a single (re-coded) row."""
    return int(sum(row[f"Q{i}A"] for i in SUBSCALE_ITEMS[subscale]))


def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Full DASS processing: validity screen → recode → scores → bands."""
    source = "real"
    if "data_source" in raw.columns and len(raw):
        source = str(raw["data_source"].iloc[0])
    df = recode_items(raw)
    df = df[validity_mask(df)].reset_index(drop=True)
    item_cols = [f"Q{i}A" for i in range(1, 43)]
    for subscale, items in SUBSCALE_ITEMS.items():
        cols = [f"Q{i}A" for i in items]
        df[f"{subscale}_score"] = df[cols].sum(axis=1, min_count=len(cols)).astype(int)
        df[f"{subscale}_band"] = df[f"{subscale}_score"].map(
            lambda s, subscale=subscale: severity_band(subscale, s)
        )
    # Demographics used by Exp 4 (gender: 1 male, 2 female, 3 other; urban 1–3).
    out = pd.DataFrame({
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "gender": df["gender"].map({1: "male", 2: "female", 3: "other"}).fillna("other"),
        "education": df["education"],
        "urban": df["urban"],
        "country": df["country"],
        "data_source": source,
    })
    for subscale in SUBSCALE_ITEMS:
        out[f"{subscale}_score"] = df[f"{subscale}_score"]
        out[f"{subscale}_band"] = df[f"{subscale}_band"]
    out[item_cols] = df[item_cols].to_numpy()  # recoded 0–3 values
    out = out[out["age"].between(10, 100)].reset_index(drop=True)
    log.info("DASS scored: %d valid respondents of %d", len(out), len(raw))
    return out


def band_to_index(band: str) -> int:
    """Ordinal index of a severity band (0 = Normal … 4 = Extremely severe)."""
    return BAND_ORDER.index(band)


def ordinal_mae(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> float:
    """Mean absolute error over ordinal band indices."""
    return float(np.mean(np.abs(np.asarray(y_true_idx) - np.asarray(y_pred_idx))))
