"""Unit tests for DASS-42 scoring, severity bands and the validity screen.

All frames are hand-built (no raw files needed) so the suite runs on a
clean CI clone. Worked examples live in ``mindsense.data.dass``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mindsense.data import dass

ITEM_COLS = [f"Q{i}A" for i in range(1, 43)]


def _mini_frame(**overrides) -> pd.DataFrame:
    """One valid raw respondent: depression items raw=4, everything else 1."""
    row: dict[str, object] = {
        "age": 30,
        "gender": 2,
        "education": 1,
        "urban": 1,
        "country": "US",
        "testelapse": 300,
        "VCL6": 0,
        "VCL9": 0,
        "VCL12": 0,
    }
    for i in range(1, 43):
        row[f"Q{i}A"] = 4 if i in dass.DEPRESSION_ITEMS else 1
    row.update(overrides)
    return pd.DataFrame([row])


def test_worked_examples_match_bands():
    for subscale, cases in dass.WORKED_EXAMPLES.items():
        for score, expected in cases:
            assert dass.severity_band(subscale, score) == expected


@pytest.mark.parametrize("subscale", sorted(dass.SEVERITY_BANDS))
def test_band_boundaries(subscale: str):
    bands = dass.SEVERITY_BANDS[subscale]
    assert dass.severity_band(subscale, 0) == "Normal"
    for i, (name, upper) in enumerate(bands):
        assert dass.severity_band(subscale, upper) == name
        if i + 1 < len(bands):
            assert dass.severity_band(subscale, upper + 1) == bands[i + 1][0]
    assert dass.severity_band(subscale, 42) == "Extremely severe"


def test_subscale_items_partition_all_42_items():
    everything = sorted(dass.DEPRESSION_ITEMS + dass.STRESS_ITEMS + dass.ANXIETY_ITEMS)
    assert everything == list(range(1, 43))
    assert all(len(v) == 14 for v in dass.SUBSCALE_ITEMS.values())


def test_band_to_index_is_ordinal():
    assert [dass.band_to_index(b) for b in dass.BAND_ORDER] == [0, 1, 2, 3, 4]
    assert dass.ordinal_mae(np.array([0, 4]), np.array([1, 2])) == 1.5


def test_recode_items_maps_1_4_to_0_3():
    raw = pd.concat([_mini_frame(), _mini_frame(Q1A=4, Q2A=3)], ignore_index=True)
    recoded = dass.recode_items(raw)
    assert recoded["Q1A"].tolist() == [0, 3]
    assert recoded["Q2A"].tolist() == [0, 2]
    assert recoded["Q3A"].tolist() == [3, 3]  # depression item raw 4 -> 3


def test_validity_mask_excludes_each_bad_case():
    valid = _mini_frame()
    bad_fake = _mini_frame(VCL6=1, VCL9=1, VCL12=1)
    bad_age = _mini_frame(age=101)
    bad_straight = _mini_frame(**dict.fromkeys(ITEM_COLS, 2))
    bad_fast = _mini_frame(testelapse=30)
    frame = pd.concat([valid, bad_fake, bad_age, bad_straight, bad_fast], ignore_index=True)
    mask = dass.validity_mask(dass.recode_items(frame))
    assert mask.tolist() == [True, False, False, False, False]


def test_score_frame_end_to_end():
    frame = pd.concat(
        [
            _mini_frame(),
            _mini_frame(VCL6=1, VCL9=1, VCL12=1),  # fake words -> dropped
            _mini_frame(age=101),  # impossible age -> dropped
        ],
        ignore_index=True,
    )
    scored = dass.score_frame(frame)
    assert len(scored) == 1
    row = scored.iloc[0]
    # depression items recoded 4-1=3 each, 14 items -> 42
    assert row["depression_score"] == 42
    assert row["depression_band"] == "Extremely severe"
    # other subscales: raw 1 -> 0
    assert row["anxiety_score"] == 0 and row["anxiety_band"] == "Normal"
    assert row["stress_score"] == 0 and row["stress_band"] == "Normal"
    assert row["gender"] == "female"
    assert row["data_source"] == "real"
    assert set(ITEM_COLS) <= set(scored.columns)
    # recoded items are 0-3
    assert scored[ITEM_COLS].to_numpy().max() <= 3


def test_score_frame_preserves_synthetic_tag():
    frame = _mini_frame().assign(data_source="synthetic")
    scored = dass.score_frame(frame)
    assert scored["data_source"].iloc[0] == "synthetic"
