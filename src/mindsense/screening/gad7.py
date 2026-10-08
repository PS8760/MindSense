"""GAD-7 anxiety questionnaire: items, scoring and severity bands.

The Generalized Anxiety Disorder-7 scale is a free, public-domain
instrument (Spitzer, Kroenke, Williams & Löwe, 2006). Score is the 0–21
sum; bands come from ``config/screening.gad7_thresholds`` ([4, 9, 14] →
minimal / mild / moderate / severe).
"""

from __future__ import annotations

from typing import Any

GAD7_ITEMS: list[str] = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen",
]

BAND_LABELS = ["Minimal", "Mild", "Moderate", "Severe"]


def _thresholds() -> list[int]:
    from mindsense.utils.io import load_config

    return list(load_config()["screening"]["gad7_thresholds"])


def band_for(total: int, thresholds: list[int] | None = None) -> str:
    """Map a 0–21 total onto its severity band."""
    cuts = thresholds if thresholds is not None else _thresholds()
    if not 0 <= total <= 21:
        raise ValueError(f"GAD-7 total must be 0–21, got {total}")
    for cut, label in zip(cuts, BAND_LABELS, strict=False):
        if total <= cut:
            return label
    return BAND_LABELS[-1]


def score_gad7(answers: list[int] | dict[int, int]) -> dict[str, Any]:
    """Score the questionnaire (seven 0–3 answers)."""
    if isinstance(answers, dict):
        values = [int(answers[i]) for i in range(1, 8)]
    else:
        values = [int(v) for v in answers]
    if len(values) != 7:
        raise ValueError(f"GAD-7 needs exactly 7 answers, got {len(values)}")
    for i, v in enumerate(values, start=1):
        if not 0 <= v <= 3:
            raise ValueError(f"GAD-7 item {i} must be 0–3, got {v}")
    total = sum(values)
    return {"total": total, "band": band_for(total), "instrument": "GAD-7"}
