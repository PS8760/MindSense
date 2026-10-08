"""PHQ-9 depression questionnaire: items, scoring and severity bands.

The Patient Health Questionnaire-9 is a free, public-domain instrument
(Kroenke, Spitzer & Williams, 2001). Scores are the plain 0–27 sum of the
nine items; band cut-points come from ``config/screening.phq9_thresholds``
([4, 9, 14, 19] → minimal / mild / moderate / moderately severe / severe).

Item 9 asks about self-harm thoughts and is the crisis trigger: any answer
> 0 raises a crisis banner immediately (Section 8, page 2).
"""

from __future__ import annotations

from typing import Any

PHQ9_ITEMS: list[str] = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have "
    "let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed — "
    "or the opposite, being so fidgety or restless that you have been "
    "moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
]

#: Likert labels shared by PHQ-9 and GAD-7 (0–3).
RESPONSE_OPTIONS: list[str] = [
    "Not at all (0)",
    "Several days (1)",
    "More than half the days (2)",
    "Nearly every day (3)",
]

ITEM9_INDEX = 8

_BAND_LABELS = ["Minimal", "Mild", "Moderate", "Moderately severe", "Severe"]


def _thresholds() -> list[int]:
    from mindsense.utils.io import load_config

    return list(load_config()["screening"]["phq9_thresholds"])


def band_for(total: int, thresholds: list[int] | None = None) -> str:
    """Map a 0–27 total onto its severity band."""
    cuts = thresholds if thresholds is not None else _thresholds()
    if not 0 <= total <= 27:
        raise ValueError(f"PHQ-9 total must be 0–27, got {total}")
    for cut, label in zip(cuts, _BAND_LABELS, strict=False):
        if total <= cut:
            return label
    return _BAND_LABELS[-1]


def score_phq9(answers: list[int] | dict[int, int]) -> dict[str, Any]:
    """Score the questionnaire.

    Parameters
    ----------
    answers:
        Nine values in 0–3, either a list in item order or a dict keyed by
        1-based item number.

    Returns
    -------
    dict with ``total`` (0–27), ``band``, ``item9`` (0–3) and
    ``crisis_triggered`` (True when item 9 > 0).
    """
    if isinstance(answers, dict):
        values = [int(answers[i]) for i in range(1, 10)]
    else:
        values = [int(v) for v in answers]
    if len(values) != 9:
        raise ValueError(f"PHQ-9 needs exactly 9 answers, got {len(values)}")
    for i, v in enumerate(values, start=1):
        if not 0 <= v <= 3:
            raise ValueError(f"PHQ-9 item {i} must be 0–3, got {v}")
    total = sum(values)
    item9 = values[ITEM9_INDEX]
    return {
        "total": total,
        "band": band_for(total),
        "item9": item9,
        "crisis_triggered": item9 > 0,
        "instrument": "PHQ-9",
    }
