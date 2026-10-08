"""Crisis detection rules shared by the app, the text pipeline and tests.

Three independent triggers (all must be surfaced to the user immediately):

1. **PHQ-9 item 9 > 0** — questionnaire self-harm item.
2. **Text classifier probability** of the suicidal class ≥
   ``config.screening.crisis.suicidal_class_threshold`` (default 0.55).
3. **Crisis lexicon** — high-precision phrases in free text
   (``config.screening.crisis.crisis_lexicon_boost``).

This module is deliberately rules-only: no model is loaded here, so the
same functions unit-test cleanly and run identically in CI (Rule 7).
"""

from __future__ import annotations

import re
from typing import Any

#: High-precision phrases (lower-case, word-boundary matched).
CRISIS_LEXICON: tuple[str, ...] = (
    "kill myself",
    "killing myself",
    "end my life",
    "want to die",
    "wanting to die",
    "wish i were dead",
    "better off dead",
    "suicidal",
    "suicide",
    "hurt myself",
    "hurting myself",
    "self harm",
    "self-harm",
    "take my own life",
    "no reason to live",
    "cant go on living",
    "can't go on living",
    "don't want to live",
    "do not want to live",
)

_PATTERNS = [re.compile(rf"\b{re.escape(phrase)}\b") for phrase in CRISIS_LEXICON]


def lexicon_hits(text: str) -> list[str]:
    """Return the crisis phrases present in ``text`` (lower-case match)."""
    lowered = text.lower()
    return [p for p, rx in zip(CRISIS_LEXICON, _PATTERNS, strict=True) if rx.search(lowered)]


def text_crisis(
    text: str,
    *,
    class_probabilities: dict[str, float] | None = None,
    suicidal_threshold: float | None = None,
) -> dict[str, Any]:
    """Combine the probability and lexicon triggers for one free-text input.

    Returns a dict with ``triggered``, ``reasons`` (human-readable list),
    ``lexicon_hits`` and ``max_suicidal_probability`` (or None).
    """
    from mindsense.utils.io import load_config

    if suicidal_threshold is None:
        cfg = load_config()["screening"]["crisis"]
        suicidal_threshold = float(cfg["suicidal_class_threshold"])
        lexicon_enabled = bool(cfg.get("crisis_lexicon_boost", True))
    else:
        lexicon_enabled = True

    reasons: list[str] = []
    hits: list[str] = []
    if lexicon_enabled:
        hits = lexicon_hits(text)
        if hits:
            reasons.append(f"crisis language detected ({', '.join(hits[:3])})")

    suicidal_prob: float | None = None
    if class_probabilities:
        for name, prob in class_probabilities.items():
            if name.lower() in {"suicidal", "suicide"}:
                suicidal_prob = max(suicidal_prob or 0.0, float(prob))
        if suicidal_prob is not None and suicidal_prob >= suicidal_threshold:
            reasons.append(
                f"suicidal-class probability {suicidal_prob:.2f} ≥ {suicidal_threshold:.2f}"
            )

    return {
        "triggered": bool(reasons),
        "reasons": reasons,
        "lexicon_hits": hits,
        "max_suicidal_probability": suicidal_prob,
        "threshold": suicidal_threshold,
    }


def questionnaire_crisis(item9: int) -> dict[str, Any]:
    """PHQ-9 item-9 trigger (any endorsement > 0)."""
    if not 0 <= int(item9) <= 3:
        raise ValueError(f"PHQ-9 item 9 must be 0–3, got {item9}")
    triggered = int(item9) > 0
    return {
        "triggered": triggered,
        "reasons": ["PHQ-9 item 9 (self-harm thoughts) endorsed"]
        if triggered
        else [],
        "item9": int(item9),
    }
