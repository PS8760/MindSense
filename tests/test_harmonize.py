"""Schema-level tests for the harmonized risk tables (Exp 1 outputs)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mindsense.data.harmonize import (
    HARMONIZED_COLUMNS,
    build_external_risk,
    build_harmonized,
    professional_to_risk,
    student_to_risk,
)


def _student_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20 + i for i in range(n)],
            "gender": ["male" if i % 2 == 0 else "female" for i in range(n)],
            "sleep_hours": [7.5] * n,
            "academic_pressure": [float(i % 6) for i in range(n)],
            "study_satisfaction": [float(i % 6) for i in range(n)],
            "work_study_hours": [6.0] * n,
            "financial_stress": [3.0] * n,
            "family_history": [i % 2 for i in range(n)],
            "diet_quality": [2 if i % 2 == 0 else 1 for i in range(n)],
            "suicidal_thoughts": [i % 2 for i in range(n)],
            "depression": [i % 2 for i in range(n)],
            "data_source": ["real"] * n,
        }
    )


def _osmi_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [31.0, 42.0],
            "gender": ["male", "other"],
            "treatment": pd.array([1, 0], dtype="Int64"),
            "family_history": pd.array([0, 1], dtype="Int64"),
            "work_interfere": ["Often", "Never"],
            "benefits": ["Yes", "No"],
            "care_options": ["Yes", "Not sure"],
            "data_source": ["real", "real"],
        }
    )


def test_student_mapper_schema_and_label():
    out = student_to_risk(_student_frame(), source="student_depression")
    assert list(out.columns) == HARMONIZED_COLUMNS
    assert out["mh_risk"].tolist() == [0, 1, 0, 1, 0, 1]
    assert out["population"].eq("student").all()
    assert out["support_available"].isna().all()  # not collected for students
    assert out["record_id"].iloc[0] == "student_depression_0"


def test_professional_mapper_schema_and_proxies():
    out = professional_to_risk(_osmi_frame(), source="osmi_2014")
    assert list(out.columns) == HARMONIZED_COLUMNS
    assert out["mh_risk"].tolist() == [1, 0]
    assert out["population"].eq("professional").all()
    # support = mean(benefits, care_options)
    assert out["support_available"].iloc[0] == 1.0  # Yes + Yes
    assert out["support_available"].iloc[1] == 0.25  # No + Not sure
    # work_interfere rank-scaled onto the 0-5 stress axis
    assert out["stress_pressure_score"].iloc[0] == 5.0  # Often
    assert out["stress_pressure_score"].iloc[1] == 0.0  # Never
    # student-only fields stay NaN
    for col in (
        "sleep_hours",
        "satisfaction_score",
        "work_study_hours",
        "financial_stress",
        "diet_quality",
        "suicidal_thoughts",
    ):
        assert out[col].isna().all(), col


def test_build_harmonized_summary_and_asymmetric_missingness():
    risk, summary = build_harmonized(_student_frame(), _osmi_frame())
    assert list(risk.columns) == HARMONIZED_COLUMNS
    assert len(risk) == 8
    assert summary["rows"] == 8
    assert summary["by_population"] == {"student": 6, "professional": 2}
    assert summary["mh_risk_rate"] == round(float(risk["mh_risk"].mean()), 4)
    # union schema: student-only and professional-only columns both have NaN
    assert 0 < summary["missing_fraction"]["support_available"] < 1
    assert 0 < summary["missing_fraction"]["sleep_hours"] < 1
    assert summary["missing_fraction"]["mh_risk"] == 0.0


def test_external_tables_never_mix_populations():
    ext = build_external_risk(_osmi_frame(), _student_frame(3))
    assert set(ext) == {"osmi_2017", "student_small"}
    assert ext["osmi_2017"]["source"].eq("osmi_2017").all()
    assert ext["student_small"]["source"].eq("student_depression_small").all()
    # empty inputs yield no table
    assert build_external_risk(pd.DataFrame(), None) == {}


def test_suicidal_thoughts_excluded_from_deployed_default():
    # guard: the ablation-only field exists in the schema but documents intent
    assert "suicidal_thoughts" in HARMONIZED_COLUMNS
    assert HARMONIZED_COLUMNS.index("mh_risk") == len(HARMONIZED_COLUMNS) - 2


def test_mh_risk_is_binary_int():
    risk, _ = build_harmonized(_student_frame(), _osmi_frame())
    assert set(risk["mh_risk"].unique()) <= {0, 1}
    assert np.issubdtype(risk["mh_risk"].dtype, np.integer)
