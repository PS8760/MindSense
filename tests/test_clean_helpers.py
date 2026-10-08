"""Pure helpers from the dataset cleaners (no raw files required)."""

from __future__ import annotations

import pandas as pd

from mindsense.data.clean import _condition_group, _snake, _yes_no, normalize_gender


def test_snake_case_column_names():
    assert _snake("Have you ever had suicidal thoughts ?") == "have_you_ever_had_suicidal_thoughts"
    assert _snake("CGPA") == "cgpa"
    assert _snake("Work/Study Hours") == "work_study_hours"
    assert _snake("Q120: Do you currently have a mental health disorder?") == (
        "q120_do_you_currently_have_a_mental_health_disorder"
    )


def test_gender_normalisation_collapses_free_text():
    cases = {
        "Male": "male",
        "femake": "female",
        "F": "female",
        "cis male": "male",
        "Non-binary": "other",
        "genderqueer": "other",
        "": "other",
        "Woman": "female",
        "agender": "other",
        "trans-female": "female",
        "guy": "male",
    }
    for raw, expected in cases.items():
        assert normalize_gender(raw) == expected, raw


def test_yes_no_handles_strings_and_numerics():
    out = _yes_no(pd.Series(["Yes", "No", 1, 0, "maybe", None]))
    assert out.tolist() == [1, 0, 1, 0, pd.NA, pd.NA]
    assert str(out.dtype) == "Int64"


def test_condition_group_matches_mental_health_only():
    assert _condition_group("Bipolar Disorde") == "bipolar"  # source typo
    assert _condition_group("Depression") == "depression"
    assert _condition_group("Anxiety") == "anxiety"
    assert _condition_group("Post-traumatic Stress Disorder") == "ptsd"
    assert _condition_group("Insomnia") == "insomnia"
    assert _condition_group("ADHD") == "adhd"
    assert _condition_group("Birth Control") is None
    assert _condition_group(None) is None
    assert _condition_group("Weight Loss") is None
