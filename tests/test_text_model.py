"""Unit tests for the Exp 6 text screen (``mindsense.models.text``).

Small synthetic corpus; no network / data/processed dependency.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from mindsense.models.text import train_text


def _frame(n: int) -> pd.DataFrame:
    rows = n // 3
    positives = [
        "i feel so anxious all the time",
        "cant stop worrying worst feeling",
        "my heart races and i panic",
    ]
    negatives = [
        "i feel fine and very relaxed today",
        "everything is going great for me",
        "calm peaceful and happy",
    ]
    text: list[str] = []
    label: list[str] = []
    for _split in ("train", "val", "test"):
        half = rows // 2
        p = (positives * ((half // len(positives)) + 1))[:half]
        q = (negatives * ((half // len(negatives)) + 1))[:half]
        text += p + q
        label += ["Anxiety"] * half + ["Normal"] * half
    split = ["train"] * rows + ["val"] * rows + ["test"] * rows
    return pd.DataFrame({"text_norm": text[:n], "label": label[:n], "split": split})


def test_train_text_ranks_models_and_reports_test_metrics(tmp_path) -> None:
    result = train_text(quick=True, data=_frame(90), out_dir=tmp_path)
    assert result["best"] in {"naive_bayes", "logistic", "linear_svc"}
    assert result["test"]["accuracy"] >= 0.0
    assert len(result["ranking"]) == 3
