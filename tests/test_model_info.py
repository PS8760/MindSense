"""Model comparison table tests — read the committed metric files."""

from __future__ import annotations

from app.components import model_info


def test_comparison_blocks_ranked_with_precision_recall() -> None:
    blocks = {b["title"]: b for b in model_info.comparison_blocks()}
    assert blocks, "expected at least one comparison block"
    image = blocks["Photo mood cue (FER-2013)"]
    rows = image["rows"]
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["key"] == image["best"]
    top = rows[0]
    assert top["accuracy"] is not None
    assert top["macro_precision"] is not None
    assert top["macro_recall"] is not None
    assert top["macro_f1"] is not None
    if image["in_use"] is not None:
        assert image["in_use"] == image["best"]


def test_regression_block_sorted_by_rmse() -> None:
    block = next(b for b in model_info.comparison_blocks() if b["kind"] == "regression")
    assert block["primary"] == "rmse_mean"
    rmses = [r["rmse_mean"] for r in block["rows"]]
    assert rmses == sorted(rmses)  # lower RMSE is better → ascending


def test_current_model_summary_lists_tasks() -> None:
    summary = model_info.current_model_summary()
    assert len(summary) == len(model_info.comparison_blocks())
    for row in summary:
        assert row["task"]
        assert row["model"]


def test_display_name_is_friendly() -> None:
    assert model_info.display_name("mlp") == "Neural network (MLP)"
    assert model_info.display_name("gradient_boosting") == "Gradient boosting"
