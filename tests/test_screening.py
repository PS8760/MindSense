"""Unit tests for PHQ-9 / GAD-7 scoring and the crisis rules (Rule 7)."""

from __future__ import annotations

import pytest

from mindsense.screening.crisis import lexicon_hits, questionnaire_crisis, text_crisis
from mindsense.screening.gad7 import score_gad7
from mindsense.screening.phq9 import score_phq9

# --------------------------------------------------------------------------- #
# PHQ-9
# --------------------------------------------------------------------------- #


def test_phq9_all_zero_is_minimal_no_crisis():
    result = score_phq9([0] * 9)
    assert result["total"] == 0
    assert result["band"] == "Minimal"
    assert result["crisis_triggered"] is False


def test_phq9_all_max_is_severe():
    result = score_phq9([3] * 9)
    assert result["total"] == 27
    assert result["band"] == "Severe"
    assert result["item9"] == 3
    assert result["crisis_triggered"] is True


def test_phq9_band_boundaries():
    # thresholds [4, 9, 14, 19] → Minimal ≤4, Mild ≤9, Moderate ≤14,
    # Moderately severe ≤19, Severe ≥20 (each item must stay within 0–3)
    assert score_phq9([1, 1, 1, 1, 0, 0, 0, 0, 0])["band"] == "Minimal"  # total 4
    assert score_phq9([2, 1, 1, 1, 0, 0, 0, 0, 0])["band"] == "Mild"  # total 5
    assert score_phq9([3, 3, 3, 0, 0, 0, 0, 0, 0])["band"] == "Mild"  # total 9
    assert score_phq9([3, 3, 3, 1, 0, 0, 0, 0, 0])["band"] == "Moderate"  # total 10
    assert score_phq9([3, 3, 3, 3, 2, 0, 0, 0, 0])["band"] == "Moderate"  # total 14
    assert score_phq9([3, 3, 3, 3, 3, 0, 0, 0, 0])["band"] == "Moderately severe"  # 15
    assert score_phq9([3, 3, 3, 3, 3, 3, 1, 0, 0])["band"] == "Moderately severe"  # 19
    assert score_phq9([3, 3, 3, 3, 3, 3, 2, 0, 0])["band"] == "Severe"  # total 20


def test_phq9_crisis_only_from_item9():
    # even a high total with item 9 = 0 must NOT raise the crisis flag
    result = score_phq9([3, 3, 3, 3, 3, 3, 3, 0, 0])
    assert result["total"] == 21
    assert result["crisis_triggered"] is False
    # conversely, a single item-9 endorsement with an otherwise empty form
    result = score_phq9([0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert result["total"] == 1
    assert result["band"] == "Minimal"
    assert result["crisis_triggered"] is True


def test_phq9_accepts_dict_answers():
    answers = dict.fromkeys(range(1, 10), 1)
    assert score_phq9(answers)["total"] == 9


def test_phq9_rejects_bad_input():
    with pytest.raises(ValueError):
        score_phq9([0] * 8)
    with pytest.raises(ValueError):
        score_phq9([0, 0, 0, 0, 0, 0, 0, 0, 4])
    with pytest.raises(ValueError):
        score_phq9([0] * 8 + [-1])


# --------------------------------------------------------------------------- #
# GAD-7
# --------------------------------------------------------------------------- #


def test_gad7_boundaries():
    # thresholds [4, 9, 14]; each item must stay within 0–3
    assert score_gad7([1, 1, 1, 1, 0, 0, 0])["band"] == "Minimal"  # total 4
    assert score_gad7([2, 1, 1, 1, 0, 0, 0])["band"] == "Mild"  # total 5
    assert score_gad7([3, 3, 3, 0, 0, 0, 0])["band"] == "Mild"  # total 9
    assert score_gad7([3, 3, 3, 1, 0, 0, 0])["band"] == "Moderate"  # total 10
    assert score_gad7([3, 3, 3, 3, 2, 0, 0])["band"] == "Moderate"  # total 14
    assert score_gad7([3, 3, 3, 3, 3, 0, 0])["band"] == "Severe"  # total 15
    assert score_gad7([3, 3, 3, 3, 3, 3, 3])["band"] == "Severe"  # total 21


def test_gad7_max_total_is_21():
    assert score_gad7([3] * 7)["total"] == 21


# --------------------------------------------------------------------------- #
# crisis rules
# --------------------------------------------------------------------------- #


def test_questionnaire_crisis_threshold():
    assert questionnaire_crisis(0)["triggered"] is False
    for value in (1, 2, 3):
        assert questionnaire_crisis(value)["triggered"] is True
    with pytest.raises(ValueError):
        questionnaire_crisis(4)


def test_lexicon_hits_are_word_bounded():
    assert "kill myself" in lexicon_hits("I want to KILL MYSELF tonight")
    assert lexicon_hits("this exam is killing me softly") == []


def test_text_crisis_lexicon_trigger():
    result = text_crisis("sometimes i wish i were dead")
    assert result["triggered"] is True
    assert result["lexicon_hits"]


def test_text_crisis_probability_trigger():
    below = text_crisis("i feel fine", class_probabilities={"suicidal": 0.30})
    assert below["triggered"] is False
    above = text_crisis("i feel fine", class_probabilities={"suicidal": 0.72})
    assert above["triggered"] is True
    assert above["max_suicidal_probability"] == 0.72


def test_text_crisis_clean_text_not_flagged():
    result = text_crisis("work has been busy but i am sleeping fine")
    assert result["triggered"] is False
    assert result["reasons"] == []
