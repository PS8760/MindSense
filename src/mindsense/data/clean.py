"""Dataset-specific cleaning into tidy, typed frames (Milestone 2, Exp 1).

Every cleaner:

1. reads its raw file (see ``mindsense.data.download`` for acquisition),
2. drops duplicates and impossible values,
3. normalises categories / text with the shared NLP preprocessor,
4. tags each row ``data_source`` (``real`` or ``synthetic``),
5. returns the tidy frame plus a quality-stats dict that the pipeline
   writes to ``reports/tables/data_quality_report.csv``.
"""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

from mindsense.data import dass as dass_score
from mindsense.nlp.preprocess import normalize_text, word_count
from mindsense.utils.io import read_table, repo_path
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.clean")

# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

_SLEEP_HOURS = {
    "Less than 5 hours": 4.0,
    "5-6 hours": 5.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 9.0,
}
_DIET_QUALITY = {"Healthy": 2, "Moderate": 1, "Unhealthy": 0}
_YES_NO = {"Yes": 1, "No": 0}


def _snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _tag_source(df: pd.DataFrame) -> pd.DataFrame:
    if "data_source" not in df.columns:
        df["data_source"] = "real"
    return df


def _yes_no(series: pd.Series) -> pd.Series:
    """Map Yes/No strings *or* 1/0 numerics to nullable integers."""
    mapped = series.map(_YES_NO)
    return (
        mapped.where(mapped.notna(), pd.to_numeric(series, errors="coerce"))
        .astype("Int64")
    )


def _raw_path(*parts: str):
    return repo_path("data", "raw", *parts)


def _counts(rows_in: int, rows_out: int, note: str = "") -> dict[str, Any]:
    """Standard quality-report row for one cleaning step."""
    return {"rows_in": rows_in, "rows_out": rows_out, "dropped": rows_in - rows_out, "note": note}


# --------------------------------------------------------------------------- #
# student depression datasets
# --------------------------------------------------------------------------- #

_STUDENT_RENAME = {
    "id": "record_id",
    "Gender": "gender",
    "Age": "age",
    "City": "city",
    "Profession": "profession",
    "Academic Pressure": "academic_pressure",
    "Work Pressure": "work_pressure",
    "CGPA": "cgpa",
    "Study Satisfaction": "study_satisfaction",
    "Job Satisfaction": "job_satisfaction",
    "Sleep Duration": "sleep_hours",
    "Dietary Habits": "diet_quality",
    "Degree": "degree",
    "Have you ever had suicidal thoughts ?": "suicidal_thoughts",
    "Work/Study Hours": "work_study_hours",
    "Study Hours": "work_study_hours",
    "Financial Stress": "financial_stress",
    "Family History of Mental Illness": "family_history",
    "Depression": "depression",
}


def clean_student_depression(*, small: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the Kaggle student-depression datasets (ranks 1 and 9)."""
    if small:
        raw = read_table(_raw_path("student_depression_small", "Depression Student Dataset.csv"))
    else:
        raw = read_table(_raw_path("student_depression", "Student Depression Dataset.csv"))
    df = raw.rename(columns=_STUDENT_RENAME).copy()
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df["gender"] = df["gender"].astype(str).str.strip().str.lower().map(
        {"male": "male", "female": "female"}
    ).fillna("other")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    n_age = int((~df["age"].between(12, 100)).sum())
    df = df[df["age"].between(12, 100)]
    df["sleep_hours"] = df["sleep_hours"].map(_SLEEP_HOURS)
    df["diet_quality"] = df["diet_quality"].map(_DIET_QUALITY)
    df["financial_stress"] = pd.to_numeric(df["financial_stress"], errors="coerce")
    df["family_history"] = _yes_no(df["family_history"])
    df["suicidal_thoughts"] = _yes_no(df["suicidal_thoughts"])
    # Depression: 0/1 in the full set, Yes/No strings in the Tier-C small set.
    df["depression"] = _yes_no(df["depression"])
    df = df.dropna(subset=["depression"])
    df["depression"] = df["depression"].astype(int)
    for col in ("academic_pressure", "study_satisfaction", "work_study_hours", "cgpa", "work_pressure", "job_satisfaction"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["population"] = "student"
    df = _tag_source(df).reset_index(drop=True)
    note = f"{n_dupes} duplicate rows removed; sleep/diet recoded to numeric; {n_age} impossible ages dropped"
    return df, _counts(len(raw), len(df), note)


# --------------------------------------------------------------------------- #
# sentiment / mental-health statements
# --------------------------------------------------------------------------- #


def clean_sentiment_mh() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the Kaggle 7-class statement corpus (rank 2)."""
    raw = read_table(_raw_path("sentiment_mh", "Combined Data.csv"))
    df = raw.dropna(subset=["statement"]).copy()
    df["statement"] = df["statement"].astype(str).str.strip()
    df = df[df["statement"] != ""]
    n_dupes = int(df.duplicated(subset=["statement"]).sum())
    df = df.drop_duplicates(subset=["statement"], keep="first")
    df["text"] = df["statement"]
    df["text_norm"] = df["text"].map(normalize_text)
    n_short = int((df["text_norm"].map(word_count) < 2).sum())
    df = df[df["text_norm"].map(word_count) >= 2]
    df["label"] = df["status"].astype(str).str.strip()
    out = df[["text", "text_norm", "label"]].reset_index(drop=True)
    out = _tag_source(out)
    note = f"{n_dupes} duplicate statements removed; {n_short} too-short after normalisation"
    return out, _counts(len(raw), len(out), note)


# --------------------------------------------------------------------------- #
# OSMI workplace surveys
# --------------------------------------------------------------------------- #

_FEMALE_HINTS = ("female", "femake", "woman", "cis female", "trans female", "trans-female")
_MALE_HINTS = ("male", "cis male", "trans male", "trans-male")
_OTHER_HINTS = (
    "other", "non-binary", "non binary", "genderqueer", "gender fluid", "genderfluid",
    "fluid", "agender", "androg", "enby", "queer", "neuter", "nb",
)


def normalize_gender(raw: Any) -> str:
    """Collapse ~50 free-text gender spellings to male/female/other."""
    v = str(raw).strip().lower()
    if v in {"", "nan", "none", "null"}:
        return "other"
    if v in {"f", "w", "female", "woman", "femme"}:
        return "female"
    if v in {"m", "male", "man", "guy", "masc"}:
        return "male"
    if any(k in v for k in _FEMALE_HINTS):
        return "female"
    if any(k in v for k in _MALE_HINTS) or re.search(r"\b(men|man)\b", v):
        return "male"
    if any(k in v for k in _OTHER_HINTS):
        return "other"
    return "other"


_OSMI_2017_RENAME = {
    "What is your age?": "age",
    "What is your gender?": "gender",
    "What country do you <strong>live</strong> in?": "country",
    "What US state or territory do you <strong>live</strong> in?": "state",
    "Have you ever sought treatment for a mental health disorder from a mental health professional?": "treatment",
    "Do you have a family history of mental illness?": "family_history",
    "If you have a mental health disorder, how often do you feel that it interferes with your work <strong>when <em>NOT</em> being treated effectively (i.e., when you are experiencing symptoms)?</strong>": "work_interfere",
    "Does your employer provide mental health benefits\xa0as part of healthcare coverage?": "benefits",
    "Do you know the options for mental health care available under your employer-provided health coverage?": "care_options",
    "<strong>Are you self-employed?</strong>": "self_employed",
    "How many employees does your company or organization have?": "no_employees",
    "If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?": "leave",
    "Do you currently have a mental health disorder?": "current_disorder",
    "Have you ever been diagnosed with a mental health disorder?": "diagnosed_disorder",
}

_OSMI_KEEP = [
    "age", "gender", "country", "state", "treatment", "family_history",
    "work_interfere", "benefits", "care_options", "self_employed",
    "no_employees", "leave", "current_disorder", "diagnosed_disorder",
]


def _osmi_core(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shared tidy step for OSMI 2014 / 2017 frames (already snake-cased)."""
    raw_len = len(df)
    df = df.drop_duplicates().copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    n_age = int((~df["age"].between(15, 100)).sum())
    df = df[df["age"].between(15, 100)]
    df["gender"] = df["gender"].map(normalize_gender)
    if "treatment" in df:
        df["treatment"] = _yes_no(df["treatment"])
        n_no_label = int(df["treatment"].isna().sum())
        df = df[df["treatment"].notna()]  # label required for mh_risk
    else:
        n_no_label = 0
    if "family_history" in df:
        df["family_history"] = _yes_no(df["family_history"])
    if "work_interfere" in df:
        df["work_interfere"] = df["work_interfere"].fillna("Not applicable")
    keep = [c for c in _OSMI_KEEP if c in df.columns]
    out = df[keep].reset_index(drop=True)
    out["source_dataset"] = source
    out = _tag_source(out)
    note = (
        f"{n_age} impossible ages dropped; {n_no_label} without treatment label dropped; "
        "work_interfere NaN -> 'Not applicable'"
    )
    return out, _counts(raw_len, len(out), note)


def clean_osmi_2014() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the OSMI 2014 workplace survey (OpenML 43674, rank 4)."""
    raw = read_table(_raw_path("osmi_2014", "osmi_2014.csv"))
    df = raw.rename(columns={c: _snake(c) for c in raw.columns})
    return _osmi_core(df, "osmi_2014")


def clean_osmi_2017() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the OSMI 2017 survey (rank 8); also accepts canonical columns.

    Accepts either the raw long question-wording columns or an already
    renamed frame (the synthetic stand-in writes canonical names).
    """
    raw = read_table(_raw_path("osmi_2017_21", "OSMI Mental Health in Tech Survey 2017.csv"))
    df = raw.rename(columns=_OSMI_2017_RENAME)
    df = df.rename(columns={c: _snake(c) for c in df.columns})
    return _osmi_core(df, "osmi_2017")


# --------------------------------------------------------------------------- #
# text corpora: dreaddit + drug reviews
# --------------------------------------------------------------------------- #


def clean_dreaddit() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean Dreaddit (rank 7): keep text + stress label, honour split."""
    train = read_table(_raw_path("dreaddit", "train-00000-of-00001.parquet"))
    test = read_table(_raw_path("dreaddit", "test-00000-of-00001.parquet"))
    train = train.assign(split="train")
    test = test.assign(split="test")
    raw = pd.concat([train, test], ignore_index=True)
    df = raw[["subreddit", "text", "label", "confidence", "split"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    n_dupes = int(df.duplicated(subset=["text"]).sum())
    df = df.drop_duplicates(subset=["text"], keep="first")
    df["text_norm"] = df["text"].map(normalize_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
    out = df.reset_index(drop=True)
    out = _tag_source(out)
    note = f"{n_dupes} cross-split duplicate texts removed (leakage guard)"
    return out, _counts(len(raw), len(out), note)


_MH_CONDITION_KEYWORDS = {
    "depression": ("depress",),
    "anxiety": ("anxiety", "anxious", "panic"),
    "bipolar": ("bipolar",),
    "insomnia": ("insomnia",),
    "adhd": ("adhd",),
    "ocd": ("ocd",),
    "ptsd": ("ptsd", "post-traumatic"),
}


def _condition_group(condition: Any) -> str | None:
    v = str(condition or "").strip().lower()
    for group, keys in _MH_CONDITION_KEYWORDS.items():
        if any(k in v for k in keys):
            return group
    return None


def clean_drug_reviews() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter/normalise Drugs.com reviews (rank 5, research-use only)."""
    train = read_table(_raw_path("drug_reviews", "drugsComTrain_raw.csv")).assign(split="train")
    test = read_table(_raw_path("drug_reviews", "drugsComTest_raw.csv")).assign(split="test")
    raw = pd.concat([train, test], ignore_index=True)
    df = raw.copy()
    df["condition_group"] = df["condition"].map(_condition_group)
    n_other = int(df["condition_group"].isna().sum())
    df = df[df["condition_group"].notna()]
    df["text"] = (
        df["review"].astype(str)
        .map(html.unescape)
        .str.strip()
        .str.strip('"')
        .str.strip()
    )
    n_empty = int((df["text"] == "").sum() + (df["text"] == "nan").sum())
    df = df[df["text"].ne("") & df["text"].ne("nan")]
    n_dupes = int(df.duplicated(subset=["text"]).sum())
    df = df.drop_duplicates(subset=["text"], keep="first")
    df["text_norm"] = df["text"].map(normalize_text)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    out = df[["uniqueID", "drugName", "condition", "condition_group", "text",
              "text_norm", "rating", "date", "usefulCount", "split"]].reset_index(drop=True)
    out = _tag_source(out)
    note = f"kept MH conditions only ({n_other} other conditions dropped); {n_empty} empty, {n_dupes} duplicate reviews removed"
    return out, _counts(len(raw), len(out), note)


# --------------------------------------------------------------------------- #
# FER2013 image manifest + DASS-42 scoring
# --------------------------------------------------------------------------- #


def clean_fer2013() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a path/emotion/split manifest for the FER2013 JPG folders."""
    root = repo_path("data", "raw", "fer2013")
    rows = []
    for split in ("train", "test"):
        for path in sorted((root / split).glob("*/*.jpg")):
            rows.append(
                {
                    "path": str(path.relative_to(repo_path())),
                    "emotion": path.parent.name,
                    "split": split,
                    "data_source": "synthetic" if path.name.startswith("synthetic_") else "real",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"no FER2013 images found under {root}")
    return out, _counts(len(out), len(out), f"{len(out)} images indexed")


def clean_dass42() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the raw DASS-42 release; drop invalid respondents (rules in data/dass.py)."""
    raw = read_table(_raw_path("dass42", "DASS_data_21.02.19", "data.csv"), sep="\t")
    scored = dass_score.score_frame(raw)
    note = (
        "invalid respondents removed: fake words>=2, age outside 10-100, "
        "straight-liners, test time <60s; items recoded 1-4 -> 0-3; "
        "bands on raw sum 0-42 (no x2 — see DECISIONS.md)"
    )
    return scored, _counts(len(raw), len(scored), note)
