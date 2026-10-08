"""DASS-42 item wording for the app's compact screener (Exp 4).

Item texts are the official questionnaire, downloaded from
https://www2.psy.unsw.edu.au/dass/Download%20files/Dass42.pdf —
the DASS is in the public domain (Lovibond & Lovibond; Psychology
Foundation of Australia). Scoring and severity bands live in
:mod:`mindsense.data.dass`; this module only supplies wording for the UI.

The compact screener shows the ≤ ``prognosis.dass_item_budget`` (14) items
selected by Experiment 4; until the prognosis artifact exists the app can
fall back to any subset passed to :func:`item_texts`.
"""

from __future__ import annotations

#: 1-based item number → statement (exact wording from the official PDF).
DASS_ITEM_TEXTS: dict[int, str] = {
    1: "I found myself getting upset by quite trivial things",
    2: "I was aware of dryness of my mouth",
    3: "I couldn't seem to experience any positive feeling at all",
    4: "I experienced breathing difficulty (e.g., excessively rapid breathing, "
    "breathlessness in the absence of physical exertion)",
    5: "I just couldn't seem to get going",
    6: "I tended to over-react to situations",
    7: "I had a feeling of shakiness (e.g., legs going to give way)",
    8: "I found it difficult to relax",
    9: "I found myself in situations that made me so anxious I was most relieved when they ended",
    10: "I felt that I had nothing to look forward to",
    11: "I found myself getting upset rather easily",
    12: "I felt that I was using a lot of nervous energy",
    13: "I felt sad and depressed",
    14: "I found myself getting impatient when I was delayed in any way "
    "(e.g., elevators, traffic lights, being kept waiting)",
    15: "I had a feeling of faintness",
    16: "I felt that I had lost interest in just about everything",
    17: "I felt I wasn't worth much as a person",
    18: "I felt that I was rather touchy",
    19: "I perspired noticeably (e.g., hands sweaty) in the absence of high "
    "temperatures or physical exertion",
    20: "I felt scared without any good reason",
    21: "I felt that life wasn't worthwhile",
    22: "I found it hard to wind down",
    23: "I had difficulty in swallowing",
    24: "I couldn't seem to get any enjoyment out of the things I did",
    25: "I was aware of the action of my heart in the absence of physical "
    "exertion (e.g., sense of heart rate increase, heart missing a beat)",
    26: "I felt down-hearted and blue",
    27: "I found that I was very irritable",
    28: "I felt I was close to panic",
    29: "I found it hard to calm down after something upset me",
    30: 'I feared that I would be "thrown" by some trivial but unfamiliar task',
    31: "I was unable to become enthusiastic about anything",
    32: "I found it difficult to tolerate interruptions to what I was doing",
    33: "I was in a state of nervous tension",
    34: "I felt I was pretty worthless",
    35: "I was intolerant of anything that kept me from getting on with what I was doing",
    36: "I felt terrified",
    37: "I could see nothing in the future to be hopeful about",
    38: "I felt that life was meaningless",
    39: "I found myself getting agitated",
    40: "I was worried about situations in which I might panic and make a fool of myself",
    41: "I experienced trembling (e.g., in the hands)",
    42: "I found it difficult to work up the initiative to do things",
}

#: Standard response instruction shown above the compact screener.
DASS_INSTRUCTION = (
    "Please read each statement and choose how much it applied to you over "
    "the past week: 0 Did not apply to me at all · 1 Applied to me to some "
    "degree, or some of the time · 2 Applied to me to a considerable degree, "
    "or a good part of the time · 3 Applied to me very much, or most of the time."
)


def item_texts(item_numbers: list[int] | None = None) -> dict[int, str]:
    """Return wording for the requested items (all 42 when ``item_numbers`` is None)."""
    if item_numbers is None:
        return dict(DASS_ITEM_TEXTS)
    missing = [n for n in item_numbers if n not in DASS_ITEM_TEXTS]
    if missing:
        raise ValueError(f"unknown DASS item number(s): {missing}")
    return {n: DASS_ITEM_TEXTS[n] for n in item_numbers}
