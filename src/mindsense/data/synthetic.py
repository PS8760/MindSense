"""Synthetic fallback generators (Section 5.6, step 4 of the acquisition ladder).

Every generator emits a seeded substitute with the *same schema* as the real
dataset so the downstream pipeline runs unchanged. Rows are tagged
``data_source="synthetic"`` later in :mod:`mindsense.data.pipeline`, and the
flag propagates to notebooks, the Lab Results page and model metadata.

Synthetic data is only ever used when the real dataset could not be obtained.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mindsense.utils.io import ensure_dir, repo_path
from mindsense.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from mindsense.data.download import DatasetResult

log = get_logger("mindsense.synthetic")
RNG_SEED = 42

STUDENT_COLUMNS = [
    "id", "Gender", "Age", "City", "Profession", "Academic Pressure", "Work Pressure",
    "CGPA", "Study Satisfaction", "Job Satisfaction", "Sleep Duration", "Dietary Habits",
    "Degree", "Have you ever had suicidal thoughts ?", "Work/Study Hours",
    "Financial Stress", "Family History of Mental Illness", "Depression",
]

SENTIMENT_LABELS = [
    "Normal", "Depression", "Suicidal", "Anxiety", "Bipolar", "Stress",
    "Personality disorder",
]

OSMI_2014_COLUMNS = [
    "Timestamp", "Age", "Gender", "Country", "state", "self_employed", "family_history",
    "treatment", "work_interfere", "no_employees", "remote_work", "tech_company",
    "benefits", "care_options", "wellness_program", "seek_help", "anonymity", "leave",
    "mental_health_consequence", "phys_health_consequence", "coworkers", "supervisor",
    "mental_health_interview", "phys_health_interview", "mental_vs_physical",
    "obs_consequence", "comments",
]

DRUG_CONDITIONS = [
    "Depression", "Anxiety", "Bipolar Disorde", "Insomnia", "ADHD", "Panic Disorder",
    "Birth Control", "Pain", "Acne",
]

# Topical filler words so generated free-text rows are unique after cleaning
# (dedupe steps key on the raw statement/review/post text).
_FILLER_WORDS = [
    "work", "school", "sleep", "night", "week", "days", "people", "feel", "home",
    "city", "news", "music", "rain", "food", "family", "friends", "morning",
    "energy", "focus", "routine", "commute", "deadline", "weekend", "habits",
]


def _unique_sentence(rng: np.random.Generator, base: str, i: int) -> str:
    """base phrasing + random topical filler + row id => unique raw text."""
    tail = " ".join(rng.choice(_FILLER_WORDS, size=4))
    return f"{base} {tail} entry {i}"


def _write_csv(df: pd.DataFrame, key: str, name: str) -> Path:
    dest = ensure_dir(repo_path("data", "raw", key)) / name
    df.to_csv(dest, index=False)
    return dest


def _student_frame(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 45, n),
        "City": rng.choice(["Mumbai", "Delhi", "Kolkata", "Pune"], n),
        "Profession": "Student",
        "Academic Pressure": rng.integers(0, 6, n),
        "Work Pressure": np.zeros(n),
        "CGPA": np.round(rng.uniform(5, 10, n), 1),
        "Study Satisfaction": rng.integers(0, 6, n),
        "Job Satisfaction": np.zeros(n),
        "Sleep Duration": rng.choice(
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], n
        ),
        "Dietary Habits": rng.choice(["Unhealthy", "Moderate", "Healthy"], n),
        "Degree": rng.choice(["B.Tech", "Class 12", "M.Sc"], n),
        "Have you ever had suicidal thoughts ?": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "Work/Study Hours": rng.integers(0, 13, n),
        "Financial Stress": rng.integers(1, 6, n).astype(float),
        "Family History of Mental Illness": rng.choice(["Yes", "No"], n),
        "Depression": rng.integers(0, 2, n),
    })


def _sentiment_frame(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    phrases = [
        "i feel so empty and tired all the time",
        "worried about everything and cannot sleep",
        "had a great day with friends today",
        "everything feels pointless lately",
        "panic attacks are getting worse",
    ]
    return pd.DataFrame({
        "Unnamed: 0": np.arange(n),
        "statement": [
            _unique_sentence(rng, phrases[i % len(phrases)], i) for i in range(n)
        ],
        "status": rng.choice(SENTIMENT_LABELS, n),
    })


def _dass_frame(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    data: dict[str, Any] = {}
    # Real DASS items are 1-4, heavily skewed low (recode -1 -> mean item ~0.8,
    # giving depression mean ~22 like the real table). Uniform 1-4 would push
    # every respondent into "Extremely severe" and leave 'Normal' with ~1
    # member, which breaks the stratified split downstream.
    item_p = [0.53, 0.25, 0.14, 0.08]
    for i in range(1, 43):
        data[f"Q{i}A"] = rng.choice([1, 2, 3, 4], n, p=item_p)
    for i in (6, 9, 12):
        data[f"VCL{i}"] = rng.integers(0, 2, n)
    data["testelapse"] = rng.integers(120, 1800, n)
    data["country"] = rng.choice(["India", "USA", "UK"], n)
    data["source"] = rng.choice([1, 2], n)
    data["education"] = rng.integers(1, 5, n)
    data["urban"] = rng.integers(1, 4, n)
    data["gender"] = rng.choice([1, 2, 3], n)
    data["age"] = rng.integers(14, 70, n)
    return pd.DataFrame(data)


def _osmi_frame(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    yes_no = lambda p=0.5: rng.choice(["Yes", "No"], n, p=[p, 1 - p])  # noqa: E731
    return pd.DataFrame({
        "Timestamp": ["2014-08-27"] * n,
        "Age": rng.integers(18, 60, n),
        "Gender": rng.choice(["Male", "Female", "Other"], n),
        "Country": rng.choice(["United States", "United Kingdom", "India"], n),
        "state": [None] * n,
        "self_employed": yes_no(0.1),
        "family_history": yes_no(0.4),
        "treatment": yes_no(0.5),
        "work_interfere": rng.choice(
            ["Often", "Sometimes", "Rarely", "Never", None], n
        ),
        "no_employees": rng.choice(["1-5", "6-25", "26-100", "100-500"], n),
        "remote_work": yes_no(0.3),
        "tech_company": yes_no(0.8),
        "benefits": rng.choice(["Yes", "No", "Don't know"], n),
        "care_options": rng.choice(["Yes", "No", "Not sure"], n),
        "wellness_program": rng.choice(["Yes", "No", "Don't know"], n),
        "seek_help": rng.choice(["Yes", "No", "Don't know"], n),
        "anonymity": rng.choice(["Yes", "No", "Don't know"], n),
        "leave": rng.choice(["Very easy", "Somewhat easy", "Somewhat hard", "Very hard"], n),
        "mental_health_consequence": rng.choice(["Yes", "Maybe", "No"], n),
        "phys_health_consequence": rng.choice(["Yes", "Maybe", "No"], n),
        "coworkers": rng.choice(["Some of them", "None of them", "All of them"], n),
        "supervisor": rng.choice(["Yes", "No", "Some of them"], n),
        "mental_health_interview": rng.choice(["Yes", "Maybe", "No"], n),
        "phys_health_interview": rng.choice(["Yes", "Maybe", "No"], n),
        "mental_vs_physical": rng.choice(["Yes", "No", "Don't know"], n),
        "obs_consequence": yes_no(0.3),
        "comments": [None] * n,
    })


def _drug_frame(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    return pd.DataFrame({
        "uniqueID": np.arange(1, n + 1),
        "drugName": rng.choice(["Sertraline", "Fluoxetine", "Escitalopram"], n),
        "condition": rng.choice(DRUG_CONDITIONS, n),
        "review": [
            _unique_sentence(
                rng, "This medication helped with my symptoms but caused some side effects.", i
            )
            for i in range(n)
        ],
        "rating": rng.integers(1, 11, n),
        "date": ["2015-01-01"] * n,
        "usefulCount": rng.integers(0, 100, n),
    })


def _small_student_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    return pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 35, n),
        "Academic Pressure": rng.integers(1, 6, n),
        "Study Satisfaction": rng.integers(1, 6, n),
        "Sleep Duration": rng.choice(["<5 hours", "5-6 hours", "7-8 hours"], n),
        "Dietary Habits": rng.choice(["Unhealthy", "Moderate", "Healthy"], n),
        "Have you ever had suicidal thoughts ?": rng.choice(["Yes", "No"], n),
        "Study Hours": rng.integers(1, 12, n),
        "Financial Stress": rng.integers(1, 6, n),
        "Family History of Mental Illness": rng.choice(["Yes", "No"], n),
        "Depression": rng.integers(0, 2, n),
    })


def _dreaddit_frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    return pd.DataFrame({
        "subreddit": rng.choice(["anxiety", "stress", "PTSD"], n),
        "post_id": [f"syn{i}" for i in range(n)],
        "sentence_range": ["(0, 100)"] * n,
        "text": [
            _unique_sentence(rng, "I have been so stressed about work and cannot cope.", i)
            for i in range(n)
        ],
        "id": np.arange(n),
        "label": rng.integers(0, 2, n),
        "confidence": rng.uniform(0.5, 1.0, n),
        "social_timestamp": rng.integers(1_500_000_000, 1_600_000_000, n),
        "social_karma": rng.integers(0, 100, n),
    })


def _osmi_2017_frame(n: int = 400) -> pd.DataFrame:
    """Subset of the 2017 wording that maps onto the harmonized schema."""
    rng = np.random.default_rng(RNG_SEED)
    return pd.DataFrame({
        "What is your age?": rng.integers(21, 60, n),
        "What is your gender?": rng.choice(["Male", "Female"], n),
        "Do you have a family history of mental illness?": rng.choice(["Yes", "No"], n),
        "Have you ever sought treatment for a mental health disorder from a mental health professional?": rng.choice(  # noqa: E501
            ["Yes", "No"], n
        ),
        "Do you currently have a mental health disorder?": rng.choice(["Yes", "No"], n),
        # Canonical names the harmonizer needs (the cleaner accepts them too).
        "work_interfere": rng.choice(["Never", "Rarely", "Sometimes", "Often", None], n),
        "benefits": rng.choice(["Yes", "No", "Don't know"], n),
        "care_options": rng.choice(["Yes", "No", "Not sure"], n),
    })


def _fer2013_images() -> list[Path]:
    """Tiny placeholder JPEGs so the image loader has a valid folder structure."""
    from PIL import Image

    rng = np.random.default_rng(RNG_SEED)
    written: list[Path] = []
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    for split in ("train", "test"):
        for cls in classes:
            dest = ensure_dir(repo_path("data", "raw", "fer2013", split, cls))
            for i in range(8):
                arr = rng.integers(0, 256, (48, 48), dtype=np.uint8)
                path = dest / f"synthetic_{i}.jpg"
                Image.fromarray(arr, mode="L").save(path)
                written.append(path)
    return written


_GENERATORS: dict[str, Any] = {
    "student_depression": lambda: [_write_csv(_student_frame(), "student_depression", "Student Depression Dataset.csv")],  # noqa: E501
    "sentiment_mh": lambda: [_write_csv(_sentiment_frame(), "sentiment_mh", "Combined Data.csv")],  # noqa: E501
    "dass42": lambda: [_dass_tsv()],
    "osmi_2014": lambda: [_write_csv(_osmi_frame(), "osmi_2014", "osmi_2014.csv")],
    "drug_reviews": lambda: [
        _write_csv(_drug_frame(), "drug_reviews", "drugsComTrain_raw.csv"),
        _write_csv(_drug_frame(500), "drug_reviews", "drugsComTest_raw.csv"),
    ],
    "fer2013": _fer2013_images,
    "dreaddit": lambda: _dreaddit_parquet(),
    "osmi_2017_21": lambda: [_write_csv(_osmi_2017_frame(), "osmi_2017_21", "OSMI Mental Health in Tech Survey 2017.csv")],  # noqa: E501
    "student_depression_small": lambda: [_write_csv(_small_student_frame(), "student_depression_small", "Depression Student Dataset.csv")],  # noqa: E501
    "druglib": lambda: [_write_csv(_drug_frame(300), "druglib", "druglibComTrain_raw.tsv")],
}


def _dass_tsv() -> Path:
    # Mirror the real download layout: the DASS zip extracts into DASS_data_21.02.19/.
    dest = ensure_dir(repo_path("data", "raw", "dass42", "DASS_data_21.02.19")) / "data.csv"
    _dass_frame().to_csv(dest, sep="\t", index=False)
    return dest


def _dreaddit_parquet() -> list[Path]:
    """Write both official split files so the cleaner finds them."""
    dest = ensure_dir(repo_path("data", "raw", "dreaddit"))
    written: list[Path] = []
    for split, n in (("train", 400), ("test", 100)):
        path = dest / f"{split}-00000-of-00001.parquet"
        _dreaddit_frame(n).to_parquet(path, index=False)
        written.append(path)
    return written


def make_synthetic(key: str, prior: "DatasetResult | None" = None) -> "DatasetResult":
    """Generate the seeded synthetic substitute for ``key`` (ladder step 4)."""
    from mindsense.data.download import DatasetResult

    gen = _GENERATORS.get(key)
    if gen is None:  # pragma: no cover - config drift guard
        return DatasetResult(key, "failed", notes="no synthetic generator defined")
    files = gen()
    from mindsense.data.download import SYNTHETIC_MARKER

    marker = ensure_dir(repo_path("data", "raw", key)) / SYNTHETIC_MARKER
    marker.write_text(f"seed={RNG_SEED}\n", encoding="utf-8")
    note = "SYNTHETIC fallback (real dataset unavailable)"
    if prior is not None:
        note = f"{note}; ladder error: {prior.notes}"
    log.warning("SYNTHETIC DATA generated for %s — will be labelled everywhere", key)
    return DatasetResult(key, "synthetic", source="synthetic", files=files, notes=note)
