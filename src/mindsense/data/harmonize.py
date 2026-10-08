"""Harmonise heterogeneous datasets onto one risk-modeling schema (Exp 1).

The prompt's Section 5.4 schema is the *union* of what the two flagship
populations actually collect — several fields exist for only one side
(students report sleep/finances/diet; professionals report workplace
support). Rows therefore carry NaN for fields their population never
answered, and models are trained per-population with a pooled baseline
for comparison (decision logged in ``docs/DECISIONS.md``).

Labels
------
* ``mh_risk`` 1/0:
  - student: ``Depression`` column (1 = depressed per dataset labelling),
  - professional: ``treatment == "Yes"`` (ever sought treatment),
* ``suicidal_thoughts`` is kept for the ablation experiment but excluded
  from the default deployed feature set (config ``features.excluded_default``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mindsense.utils.logging import get_logger

log = get_logger("mindsense.harmonize")

HARMONIZED_COLUMNS = [
    "record_id",
    "source",
    "population",
    "age",
    "gender",
    "sleep_hours",
    "stress_pressure_score",
    "satisfaction_score",
    "work_study_hours",
    "financial_stress",
    "family_history",
    "support_available",
    "diet_quality",
    "suicidal_thoughts",
    "mh_risk",
    "data_source",
]

#: work_interfere categories mapped to the 0-5 stress scale by rank scaling
#: (Never < Rarely < Sometimes < Often), so order is preserved.
_WORK_INTERFERE_TO_STRESS = {
    "Never": 0.0,
    "Rarely": round(5 / 3, 2),
    "Sometimes": round(10 / 3, 2),
    "Often": 5.0,
}
_BENEFIT_TO_SUPPORT = {"Yes": 1.0, "Don't know": 0.5, "No": 0.0}
_CARE_TO_SUPPORT = {"Yes": 1.0, "Not sure": 0.5, "No": 0.0}


def _family(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def student_to_risk(student: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Map a cleaned student-depression frame onto the harmonized schema."""
    out = pd.DataFrame(
        {
            "record_id": [f"{source}_{i}" for i in range(len(student))],
            "source": source,
            "population": "student",
            "age": student["age"].astype(float),
            "gender": student["gender"],
            "sleep_hours": student.get("sleep_hours"),
            "stress_pressure_score": student.get("academic_pressure"),
            "satisfaction_score": student.get("study_satisfaction"),
            "work_study_hours": student.get("work_study_hours"),
            "financial_stress": student.get("financial_stress"),
            "family_history": _family(student["family_history"]),
            "support_available": np.nan,  # not collected for students
            "diet_quality": student.get("diet_quality"),
            "suicidal_thoughts": _family(student["suicidal_thoughts"]),
            "mh_risk": student["depression"].astype(int),
            "data_source": student.get("data_source", "real"),
        }
    )
    return out[HARMONIZED_COLUMNS]


def professional_to_risk(osmi: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Map a cleaned OSMI survey frame onto the harmonized schema."""
    support_parts = []
    if "benefits" in osmi:
        support_parts.append(osmi["benefits"].map(_BENEFIT_TO_SUPPORT))
    if "care_options" in osmi:
        support_parts.append(osmi["care_options"].map(_CARE_TO_SUPPORT))
    if support_parts:
        support = pd.concat(support_parts, axis=1).mean(axis=1, skipna=True)
    else:  # pragma: no cover - schema guard
        support = pd.Series(np.nan, index=osmi.index)

    out = pd.DataFrame(
        {
            "record_id": [f"{source}_{i}" for i in range(len(osmi))],
            "source": source,
            "population": "professional",
            "age": osmi["age"].astype(float),
            "gender": osmi["gender"],
            "sleep_hours": np.nan,
            "stress_pressure_score": osmi["work_interfere"].map(_WORK_INTERFERE_TO_STRESS),
            "satisfaction_score": np.nan,
            "work_study_hours": np.nan,
            "financial_stress": np.nan,
            "family_history": _family(osmi["family_history"]),
            "support_available": support,
            "diet_quality": np.nan,
            "suicidal_thoughts": pd.array([pd.NA] * len(osmi), dtype="Int64"),
            "mh_risk": osmi["treatment"].astype(int),
            "data_source": osmi.get("data_source", "real"),
        }
    )
    return out[HARMONIZED_COLUMNS]


def build_harmonized(
    student: pd.DataFrame,
    osmi_2014: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Concatenate the two in-distribution populations into the risk table.

    ``student_depression_small`` is deliberately excluded — config assigns it
    to experiments 6/8 only (external shift check, Tier C).
    """
    df = pd.concat(
        [
            student_to_risk(student, source="student_depression"),
            professional_to_risk(osmi_2014, source="osmi_2014"),
        ],
        ignore_index=True,
    )
    summary = {
        "rows": len(df),
        "by_population": df["population"].value_counts().to_dict(),
        "by_source": df["source"].value_counts().to_dict(),
        "mh_risk_rate": round(float(df["mh_risk"].mean()), 4),
        "missing_fraction": {
            col: round(float(df[col].isna().mean()), 3) for col in HARMONIZED_COLUMNS
        },
    }
    log.info(
        "harmonized risk table: %d rows (%s)", len(df), summary["by_population"]
    )
    return df, summary


def build_external_risk(osmi_2017: pd.DataFrame, student_small: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Assemble the two external-shift tables (never mixed into training)."""
    external: dict[str, pd.DataFrame] = {}
    if osmi_2017 is not None and len(osmi_2017):
        external["osmi_2017"] = professional_to_risk(osmi_2017, source="osmi_2017")
    if student_small is not None and len(student_small):
        external["student_small"] = student_to_risk(student_small, source="student_depression_small")
    return external
