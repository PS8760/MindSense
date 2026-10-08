"""Experiment 1 pipeline: clean → harmonize → split → reports (Milestone 2).

Run via ``python scripts/run_pipeline.py`` (or ``make data`` after the
download step). Outputs:

* ``data/processed/*.parquet`` — tidy tables; modeling tables carry a
  ``split`` column (stratified 70/15/15, seed 42; official splits kept
  where the source provides one).
* ``reports/tables/data_quality_report.csv`` — per-step row counts/notes
* ``reports/tables/text_leakage_report.json`` — cross-corpus duplicates
* ``reports/metrics/exp01_pipeline.json`` — machine-readable summary
* ``docs/DATA_DICTIONARY.md`` — generated dictionary + Mermaid lineage

Dataset inventory (sources, hashes, licences) is written separately by
``scripts/download_data.py``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mindsense.data.clean import (
    clean_dass42,
    clean_dreaddit,
    clean_drug_reviews,
    clean_fer2013,
    clean_osmi_2014,
    clean_osmi_2017,
    clean_sentiment_mh,
    clean_student_depression,
)
from mindsense.data.features import leakage_report, stratified_splits
from mindsense.data.harmonize import build_external_risk, build_harmonized
from mindsense.utils.io import ensure_dir, load_config, repo_path, save_json, write_parquet
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.pipeline")

SEED = 42

#: dataset key -> (cleaner, processed filename)
CLEANERS = {
    "student_depression": (lambda: clean_student_depression(small=False), "student_depression.parquet"),
    "student_depression_small": (lambda: clean_student_depression(small=True), "student_depression_small.parquet"),
    "sentiment_mh": (clean_sentiment_mh, "sentiment_mh.parquet"),
    "dass42": (clean_dass42, "dass42.parquet"),
    "osmi_2014": (clean_osmi_2014, "osmi_2014.parquet"),
    "osmi_2017_21": (clean_osmi_2017, "osmi_2017.parquet"),
    "drug_reviews": (clean_drug_reviews, "drug_reviews.parquet"),
    "fer2013": (clean_fer2013, "fer2013_manifest.parquet"),
    "dreaddit": (clean_dreaddit, "dreaddit.parquet"),
}

CORE = {"student_depression", "osmi_2014", "sentiment_mh", "dass42"}

#: Modelling tables get a stratified ``split`` column; official splits are
#: preserved for drug_reviews/dreaddit (their source ships one).
SPLIT_STRATA = {
    "risk_harmonized.parquet": ("mh_risk", "population"),
    "sentiment_mh.parquet": ("label",),
    "dass42.parquet": ("depression_band",),
}

DATA_QUALITY_NOTES = {
    "student_depression": [
        "Sleep/Diet 'Others' placeholders -> NaN; sleep buckets mapped to midpoints (4/5.5/7.5/9 h).",
        "3 rows with missing Financial Stress kept as NaN (filled at fit time, never here).",
        "Suicidal-thoughts field retained for ablation only; excluded from the deployed model.",
        "Near-constant Work Pressure / Job Satisfaction kept in the clean table, dropped when modeling.",
    ],
    "osmi_2014": [
        "Ages outside 15-100 dropped (source contains absurd outliers).",
        "Gender free-text normalised to male/female/other (49 raw spellings observed).",
        "work_interfere NaN -> 'Not applicable' (employees without a condition).",
        "'treatment' is a proxy label (has ever sought treatment), not a diagnosis.",
    ],
    "sentiment_mh": [
        "362 null statements, 1608 duplicate statements and 230 too-short rows removed before splitting.",
        "Text normalised: HTML unescaped, URLs/handles removed, emoji -> word tokens, lowercased.",
        "Class imbalance retained; macro-F1 used for evaluation (Exp 5/7).",
        "2819 statements are verbatim Dreaddit posts (the corpus is Reddit-scraped) - Exp 7 must "
        "drop overlap rows from training before evaluating on Dreaddit (text_leakage_report.json).",
    ],
    "dass42": [
        "Items recoded 1-4 -> 0-3 per the Open Psychometrics codebook.",
        "Validity screen: >=2 fake VCL words, age outside 10-100, straight-liners, test time <60 s.",
        "Severity bands applied to the RAW 14-item sum 0-42; the DASS-21 x2 rule is not used "
        "(the two cut-off tables align once doubled - see DECISIONS.md).",
        "Observed distribution is skewed toward severe bands: self-selected sample of people "
        "who chose to take an online depression test - not a population prevalence estimate.",
    ],
    "drug_reviews": [
        "Filtered to mental-health conditions (depression/anxiety/bipolar/insomnia/adhd/ocd/ptsd); "
        "'Bipolar Disorde' typo matched by prefix.",
        "Research-use only, no redistribution: processed table stays git-ignored; only aggregates published.",
        "Official train/test split from the source files preserved.",
    ],
    "dreaddit": [
        "Official train/test split preserved; cross-split duplicate texts removed (leakage guard).",
        "LIWC/engineering features dropped - only text + label kept for Exp 7.",
    ],
    "risk_harmonized": [
        "Population-asymmetric schema: students have sleep/finances/diet, professionals have "
        "workplace support; the other side is NaN by design (per-population models + pooled baseline).",
        "Stratified 70/15/15 split on mh_risk x population, seed 42, written into the split column.",
    ],
    "osmi_2017_shift": [
        "External temporal-shift evaluation only (Exp 6); never mixed into training.",
    ],
    "student_depression_small": [
        "Tier-C external sanity check (Exp 6); never mixed into training.",
    ],
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _present(key: str) -> bool:
    raw = repo_path("data", "raw", key)
    return raw.exists() and any(p for p in raw.rglob("*") if p.is_file())


def _assign_split(df: pd.DataFrame, strata_cols: tuple[str, ...]) -> pd.DataFrame:
    """Add a stratified train/val/test ``split`` column (seed 42)."""
    out = df.copy()
    splits = stratified_splits(out, strata_cols=strata_cols, seed=SEED)
    col = np.empty(len(out), dtype=object)
    for name, idx in splits.items():
        col[idx] = name
    out["split"] = col
    return out


def _table_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_pct": round(float(df.isna().mean().mean()) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "split" in df.columns:
        summary["splits"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    if "data_source" in df.columns:
        summary["data_source"] = {str(k): int(v) for k, v in df["data_source"].value_counts().items()}
    return summary


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #


def clean_all() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Clean every present raw dataset and write its processed parquet."""
    frames: dict[str, pd.DataFrame] = {}
    quality: list[dict[str, Any]] = []
    processed = ensure_dir(repo_path("data", "processed"))
    for key, (cleaner, filename) in CLEANERS.items():
        if not _present(key):
            quality.append({"dataset": key, "stage": "clean", "rows_in": 0, "rows_out": 0,
                            "dropped": 0, "note": "raw data not present - skipped"})
            log.warning("raw data missing for %s - skipped", key)
            continue
        try:
            df, stats = cleaner()
        except Exception as exc:  # noqa: BLE001 - report and continue
            quality.append({"dataset": key, "stage": "clean", "rows_in": 0, "rows_out": 0,
                            "dropped": 0, "note": f"FAILED: {exc}"})
            log.exception("cleaning %s failed", key)
            continue
        write_parquet(df, processed / filename)
        frames[key] = df
        quality.append({"dataset": key, "stage": "clean", **stats})
        log.info("cleaned %-24s %6d -> %6d rows", key, stats["rows_in"], stats["rows_out"])
    return frames, quality


def build_modeling_tables(
    frames: dict[str, pd.DataFrame], quality: list[dict[str, Any]]
) -> dict[str, pd.DataFrame]:
    """Harmonize the risk table, build external tables, attach splits."""
    processed = repo_path("data", "processed")
    tables: dict[str, pd.DataFrame] = {}

    missing = CORE - set(frames)
    if missing:
        raise SystemExit(f"core datasets missing, cannot build risk table: {sorted(missing)}")

    risk, summary = build_harmonized(frames["student_depression"], frames["osmi_2014"])
    risk = _assign_split(risk, SPLIT_STRATA["risk_harmonized.parquet"])
    write_parquet(risk, processed / "risk_harmonized.parquet")
    save_json(summary, processed / "risk_harmonized_summary.json")
    tables["risk_harmonized"] = risk
    quality.append({"dataset": "risk_harmonized", "stage": "harmonize", "rows_in": len(risk),
                    "rows_out": len(risk), "dropped": 0,
                    "note": f"mh_risk rate {summary['mh_risk_rate']}; {summary['by_population']}"})

    external = build_external_risk(
        frames.get("osmi_2017_21", pd.DataFrame()),
        frames.get("student_depression_small"),
    )
    external_names = {"osmi_2017": "osmi_2017_shift.parquet",
                      "student_small": "student_small_external_risk.parquet"}
    for name, ext in external.items():
        write_parquet(ext, processed / external_names[name])
        tables[f"external_{name}"] = ext
        quality.append({"dataset": name, "stage": "harmonize", "rows_in": len(ext),
                        "rows_out": len(ext), "dropped": 0,
                        "note": "external evaluation only (Exp 6), never trained on"})

    # Modeling tables that need a stratified split column.
    for filename, strata in SPLIT_STRATA.items():
        if filename == "risk_harmonized.parquet":
            continue  # already handled above
        key = {"sentiment_mh.parquet": "sentiment_mh", "dass42.parquet": "dass42"}[filename]
        if key not in frames:
            continue
        df = _assign_split(frames[key], strata)
        write_parquet(df, processed / filename)
        frames[key] = df
        quality.append({"dataset": key, "stage": "split", "rows_in": len(df), "rows_out": len(df),
                        "dropped": 0, "note": f"stratified by {'|'.join(strata)}"})

    # Official splits surfaced as a column for the text corpora.
    for key in ("drug_reviews", "dreaddit"):
        if key in frames and "split" not in frames[key].columns:
            frames[key] = frames[key].assign(split="train")

    if "sentiment_mh" in frames and "dreaddit" in frames:
        leak = leakage_report(frames["sentiment_mh"], frames["dreaddit"])
        save_json(leak, repo_path("reports", "tables", "text_leakage_report.json"))
        overlap = leak["sentiment_vs_dreaddit"]["overlap"]
        quality.append({"dataset": "text_leakage", "stage": "leakage_check", "rows_in": 0,
                        "rows_out": 0, "dropped": overlap,
                        "note": "exact normalised-text overlap between corpora"})
        if overlap:
            log.warning("cross-corpus text overlap: %d exact matches", overlap)
    return tables


def quality_report(
    frames: dict[str, pd.DataFrame], quality: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write the CSV row-count report and the JSON summary."""
    tables_dir = ensure_dir(repo_path("reports", "tables"))
    pd.DataFrame(quality).to_csv(tables_dir / "data_quality_report.csv", index=False)

    inventory_summary: dict[str, Any] = {}
    inv_path = tables_dir / "data_inventory.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        inventory_summary = {
            "datasets_configured": int(len(inv)),
            "datasets_present": int((inv["access_status"] != "failed").sum()),
            "synthetic_used": int((inv["data_source"] == "synthetic").sum()),
        }

    report: dict[str, Any] = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "split_ratios": [0.70, 0.15, 0.15],
        "tables": {name: _table_summary(df) for name, df in frames.items()},
        "inventory_summary": inventory_summary,
        "notes": DATA_QUALITY_NOTES,
    }
    save_json(report, repo_path("reports", "metrics", "exp01_pipeline.json"))
    log.info("quality report -> reports/tables/data_quality_report.csv")
    return report


# --------------------------------------------------------------------------- #
# generated documentation
# --------------------------------------------------------------------------- #

DICTIONARY_SPEC: dict[str, dict[str, str]] = {
    "risk_harmonized.parquet": {
        "record_id": "str | synthetic '<source>_<row>' id | pipeline | not present in raw data",
        "source": "category | originating dataset key | pipeline | provenance",
        "population": "category student/professional | source population | derived | constant per source",
        "age": "float 15-100 | respondent age (years) | Student/OSMI | invalid ages dropped at clean stage",
        "gender": "category male/female/other | gender | Student/OSMI | 49 free-text spellings normalised",
        "sleep_hours": "float 4-9 | nightly sleep bucket midpoint | Student | 'Others' -> NaN; NaN for professionals",
        "stress_pressure_score": "float 0-5 | academic pressure (student) / work-interfere rank scaled (professional) | Student/OSMI | rank Never<Rarely<Sometimes<Often scaled to 0-5",
        "satisfaction_score": "float 0-5 | study satisfaction | Student | NaN for professionals",
        "work_study_hours": "float 0-13 | daily work/study hours | Student | NaN for professionals",
        "financial_stress": "float 1-5 | financial stress rating | Student | non-numeric -> NaN; NaN for professionals",
        "family_history": "Int64 0/1 | family history of mental illness | Student/OSMI | Yes/No -> 1/0",
        "support_available": "float 0-1 | employer benefits + care-option awareness mean | OSMI | {Yes:1, Don't know/Not sure:0.5, No:0}; NaN for students",
        "diet_quality": "Int64 0-2 | 0 unhealthy .. 2 healthy | Student | 'Others' -> NaN; NaN for professionals",
        "suicidal_thoughts": "Int64 0/1 | ablation-only sensitive field | Student | excluded from deployed model; NaN for professionals",
        "mh_risk": "int 0/1 | PROXY target: student Depression label / professional ever-sought-treatment | Student/OSMI | not a diagnosis",
        "data_source": "category real/synthetic | provenance tag | pipeline | set by acquisition ladder",
        "split": "category train/val/test | stratified 70/15/15 by mh_risk x population | pipeline | seed 42",
    },
    "dass42.parquet": {
        "age": "float 10-100 | respondent age | Open Psychometrics | impossible ages dropped",
        "gender": "category male/female/other | gender | Open Psychometrics | 1/2/3 codes mapped",
        "education": "int 1-4 | highest education level | Open Psychometrics | 1 < HS .. 4 graduate",
        "urban": "int 1-3 | childhood area rural->urban | Open Psychometrics | as collected",
        "country": "str | ISO country code of connection | Open Psychometrics | as collected",
        "data_source": "category real/synthetic | provenance tag | pipeline | real for the official release",
        "depression_score": "int 0-42 | DASS depression subscale | derived | raw sum of 14 items after 0-3 recode (no x2)",
        "anxiety_score": "int 0-42 | DASS anxiety subscale | derived | raw sum of 14 items",
        "stress_score": "int 0-42 | DASS stress subscale | derived | raw sum of 14 items",
        "*_band": "category Normal..Extremely severe | severity band | derived | Lovibond & Lovibond cut-offs",
        "Q1A..Q42A": "int 0-3 | individual item response | Open Psychometrics | 1-4 -> 0-3 recode",
        "split": "category train/val/test | stratified by depression_band | pipeline | seed 42",
    },
    "sentiment_mh.parquet": {
        "text": "str | original statement | Kaggle sentiment_mh | trimmed only",
        "text_norm": "str | normalised statement | pipeline | HTML unescape, URL/handle strip, emoji->token, lowercase",
        "label": "category (7) | Normal/Depression/Suicidal/Anxiety/Bipolar/Stress/Personality disorder | Kaggle | original status labels",
        "data_source": "category real/synthetic | provenance tag | pipeline | real",
        "split": "category train/val/test | stratified 70/15/15 by label | pipeline | seed 42; split AFTER dedup",
    },
    "drug_reviews.parquet": {
        "uniqueID": "int | source row id | UCI/Kaggle mirror | as collected",
        "drugName": "str | brand name | UCI | as collected",
        "condition": "str | verbatim condition | UCI | as collected (incl. 'Bipolar Disorde' typo)",
        "condition_group": "category depression/anxiety/bipolar/insomnia/adhd/ocd/ptsd | mental-health group | derived | keyword match; other conditions dropped",
        "text": "str | review body | UCI | HTML entities unescaped, quotes stripped",
        "text_norm": "str | normalised review | pipeline | same recipe as sentiment_mh",
        "rating": "float 1-10 | patient rating | UCI | as collected",
        "date": "str | review date | UCI | as collected (DD-Mon-YY)",
        "usefulCount": "int | helpfulness votes | UCI | as collected",
        "split": "category train/test | official source split | UCI | preserved, not re-stratified",
        "data_source": "category real/synthetic | provenance tag | pipeline | real",
    },
    "dreaddit.parquet": {
        "subreddit": "str | source subreddit | Dreaddit (HF) | as collected",
        "text": "str | original Reddit post | Dreaddit | trimmed",
        "text_norm": "str | normalised post | pipeline | same recipe as sentiment_mh",
        "label": "int 0/1 | stress-related label | Dreaddit | Turcan & McKeown 2019",
        "confidence": "float | annotator confidence | Dreaddit | as collected",
        "split": "category train/test | official source split | Dreaddit | preserved; cross-split dupes removed",
        "data_source": "category real/synthetic | provenance tag | pipeline | real",
    },
    "student_depression.parquet": {
        "record_id": "int | source row id | Kaggle | as collected",
        "gender": "category male/female/other | gender | Kaggle | case-normalised",
        "age": "float 12-100 | age in years | Kaggle | impossible ages dropped",
        "city": "str | city | Kaggle | kept for EDA only (high cardinality, not a model feature)",
        "profession": "str | occupation | Kaggle | kept for EDA only",
        "academic_pressure": "int 0-5 | self-rated academic pressure | Kaggle | becomes stress_pressure_score",
        "work_pressure": "int | work pressure | Kaggle | near-constant 0 - dropped when modeling",
        "cgpa": "float | CGPA | Kaggle | as collected",
        "study_satisfaction": "int 0-5 | study satisfaction | Kaggle | becomes satisfaction_score",
        "job_satisfaction": "int | job satisfaction | Kaggle | near-constant 0 - dropped when modeling",
        "sleep_hours": "float 4-9 | sleep bucket midpoint | Kaggle | 'Others' -> NaN",
        "diet_quality": "Int64 0-2 | 0 unhealthy .. 2 healthy | Kaggle | 'Others' -> NaN",
        "degree": "str | degree pursued | Kaggle | EDA only",
        "suicidal_thoughts": "Int64 0/1 | suicidal ideation history | Kaggle | ablation-only field",
        "work_study_hours": "float 0-13 | daily hours | Kaggle | as collected",
        "financial_stress": "float 1-5 | financial stress | Kaggle | non-numeric -> NaN",
        "family_history": "Int64 0/1 | family history | Kaggle | Yes/No -> 1/0",
        "depression": "int 0/1 | TARGET label | Kaggle | dataset's labelling, not a clinical diagnosis",
        "population": "category student | population tag | pipeline | constant",
        "data_source": "category real/synthetic | provenance tag | pipeline | real",
    },
    "student_depression_small.parquet": {
        "gender": "category male/female/other | gender | Kaggle Tier C | case-normalised",
        "age": "float 12-100 | age in years | Kaggle Tier C | impossible ages dropped",
        "academic_pressure": "int 0-5 | self-rated academic pressure | Kaggle Tier C | as collected",
        "study_satisfaction": "int 0-5 | study satisfaction | Kaggle Tier C | as collected",
        "sleep_hours": "float 4-9 | sleep bucket midpoint | Kaggle Tier C | 'Others' -> NaN",
        "diet_quality": "Int64 0-2 | 0 unhealthy .. 2 healthy | Kaggle Tier C | 'Others' -> NaN",
        "suicidal_thoughts": "Int64 0/1 | suicidal ideation history | Kaggle Tier C | ablation-only field",
        "work_study_hours": "float | daily study hours ('Study Hours' renamed) | Kaggle Tier C | as collected",
        "financial_stress": "float 1-5 | financial stress | Kaggle Tier C | non-numeric -> NaN",
        "family_history": "Int64 0/1 | family history | Kaggle Tier C | Yes/No -> 1/0",
        "depression": "int 0/1 | TARGET label (Yes/No strings recoded) | Kaggle Tier C | dataset's labelling, not a clinical diagnosis",
        "population": "category student | population tag | pipeline | constant",
        "data_source": "category real/synthetic | provenance tag | pipeline | real",
    },
    "osmi_2014.parquet": {
        "age": "float 15-100 | age in years | OSMI 2014 | impossible ages dropped",
        "gender": "category male/female/other | gender | OSMI 2014 | free-text normalised",
        "country": "str | country | OSMI 2014 | as collected",
        "state": "str | US state | OSMI 2014 | as collected",
        "treatment": "Int64 0/1 | ever sought treatment | OSMI 2014 | becomes mh_risk for professionals",
        "family_history": "Int64 0/1 | family history | OSMI 2014 | Yes/No -> 1/0",
        "work_interfere": "category Never/Rarely/Sometimes/Often/Not applicable | work interference | OSMI 2014 | NaN -> Not applicable",
        "benefits": "category Yes/No/Don't know | employer MH benefits | OSMI 2014 | feeds support_available",
        "care_options": "category Yes/No/Not sure | care-option awareness | OSMI 2014 | feeds support_available",
        "self_employed": "category Yes/No | self employment | OSMI 2014 | as collected",
        "no_employees": "str | company size band | OSMI 2014 | as collected",
        "leave": "category | ease of MH leave | OSMI 2014 | as collected",
        "source_dataset": "category osmi_2014/osmi_2017 | survey wave | pipeline | provenance",
        "data_source": "category real/synthetic | provenance tag | pipeline | real",
    },
    "osmi_2017.parquet": {
        "current_disorder": "category Yes/No | self-reported current disorder | OSMI 2017 | long wording mapped to short names",
        "diagnosed_disorder": "category Yes/No | ever diagnosed | OSMI 2017 | long wording mapped to short names",
        "(other columns)": "as osmi_2014.parquet | OSMI 2017 | same tidy schema after rename",
    },
    "osmi_2017_shift.parquet": {
        "(columns)": "as risk_harmonized.parquet | OSMI 2017 | harmonized external table, Exp 6 temporal-shift evaluation only",
    },
    "student_small_external_risk.parquet": {
        "(columns)": "as risk_harmonized.parquet | Kaggle Tier C | harmonized external table, Exp 6 sanity check only",
    },
    "fer2013_manifest.parquet": {
        "path": "str | relative path to 48x48 JPG | FER2013 (Kaggle) | repo-relative for loaders",
        "emotion": "category (7) | angry/disgust/fear/happy/sad/surprise/neutral | FER2013 | folder name",
        "split": "category train/test | official split | FER2013 | preserved",
        "data_source": "category real/synthetic | provenance tag | pipeline | real for the Kaggle release",
    },
}

_LINEAGE = """```mermaid
flowchart TD
  subgraph raw["data/raw (git-ignored)"]
    SD["student_depression.csv"]
    SS["student_depression_small.csv"]
    SM["sentiment_mh/Combined Data.csv"]
    D42["dass42/data.csv"]
    O14["osmi_2014.csv"]
    O17["osmi_2017.csv"]
    DR["drugsCom*_raw.csv"]
    DDT["dreaddit/*.parquet"]
    FER["fer2013/train|test/*.jpg"]
  end
  subgraph clean["mindsense.data.clean"]
    C1["clean_student_depression"]
    C2["clean_sentiment_mh"]
    C3["score_frame (validity + bands)"]
    C4["clean_osmi_2014 / _2017"]
    C5["clean_drug_reviews (MH filter)"]
    C6["clean_dreaddit (dedupe)"]
    C7["clean_fer2013 (manifest)"]
  end
  subgraph proc["data/processed (git-ignored)"]
    P1["student_depression.parquet"]
    P4["osmi_2014.parquet"]
    HR["risk_harmonized.parquet (+split)"]
    EXT["osmi_2017_shift.parquet / student_small_external_risk.parquet"]
    TXT["sentiment_mh.parquet (+split)"]
    DAS["dass42.parquet (+split)"]
    DRP["drug_reviews.parquet (official split)"]
    DDT2["dreaddit.parquet (official split)"]
    FERM["fer2013_manifest.parquet"]
  end
  subgraph out["reports + docs (committed)"]
    INV["data_inventory.csv"]
    QR["data_quality_report.csv"]
    JSON1["exp01_pipeline.json"]
    LK["text_leakage_report.json"]
    DD["DATA_DICTIONARY.md"]
  end
  SD --> C1 --> P1
  SS --> C1
  O14 --> C4 --> P4
  P1 --> HR
  P4 --> HR
  SM --> C2 --> TXT
  D42 --> C3 --> DAS
  O17 --> C4
  DR --> C5 --> DRP
  DDT --> C6 --> DDT2
  FER --> C7 --> FERM
  O17 --> EXT
  SS --> EXT
  HR --> DD
```"""


def write_data_dictionary() -> Path:
    """Generate ``docs/DATA_DICTIONARY.md`` (dictionary + lineage diagram)."""
    lines = [
        "# Data Dictionary",
        "",
        "> Generated by `python scripts/run_pipeline.py` — do not edit by hand;",
        "> update `DICTIONARY_SPEC` in `src/mindsense/data/pipeline.py` instead.",
        "",
        "All processed tables live in `data/processed/` (git-ignored; licences "
        "prohibit redistribution for several sources) and are regenerated by "
        "`make data`. Rows carry `data_source = real|synthetic`. Raw sources, "
        "hashes and licences: `reports/tables/data_inventory.csv`.",
        "",
    ]
    for table, cols in DICTIONARY_SPEC.items():
        lines += [
            f"## `data/processed/{table}`",
            "",
            "| Column | Type / range | Description | Source | Transformation |",
            "|---|---|---|---|---|",
        ]
        for col, desc in cols.items():
            parts = desc.split(" | ", 3)
            if len(parts) != 4:  # pragma: no cover - spec guard
                parts = (parts + ["", "", ""])[:4]
            lines.append(f"| `{col}` | " + " | ".join(parts) + " |")
        lines.append("")
    lines += ["## Data lineage", "", _LINEAGE, ""]
    path = repo_path("docs", "DATA_DICTIONARY.md")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote docs/DATA_DICTIONARY.md")
    return path


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def run_pipeline() -> dict[str, Any]:
    """Execute the full Exp 1 pipeline and write every artifact."""
    load_config()  # validate config parses before doing work
    frames, quality = clean_all()
    tables = build_modeling_tables(frames, quality)
    report = quality_report({**frames, **tables}, quality)
    write_data_dictionary()
    log.info(
        "pipeline complete: %s",
        {k: len(v) for k, v in {**frames, **tables}.items()},
    )
    return report


if __name__ == "__main__":  # pragma: no cover
    run_pipeline()
