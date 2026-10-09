"""Model comparison tables + "currently in use" reporting for the frontend.

Reads the saved experiment metrics under ``reports/metrics/`` (never edited by
hand), ranks each experiment's models by its primary metric, and marks which
model the app actually serves today (a local artifact / deployed ONNX, or the
Groq AI backend when no artifact ships).
"""

from __future__ import annotations

from typing import Any

from mindsense import inference
from mindsense.utils.io import load_json, repo_path

#: Raw estimator keys → friendly display names.
_DISPLAY = {
    "logistic": "Logistic regression",
    "logreg": "Logistic regression",
    "linear": "Linear regression",
    "random_forest": "Random forest",
    "forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "mlp": "Neural network (MLP)",
    "naive_bayes": "Naive Bayes",
    "linear_svc": "Linear SVM",
}

#: (file, title, task, primary metric, higher-is-better, inference status key, kind)
_SPECS = (
    (
        "exp03_image",
        "Photo mood cue (FER-2013)",
        "7-class facial emotion",
        "macro_f1",
        True,
        "face",
        "classification",
    ),
    (
        "exp04_tabular",
        "Lifestyle risk screen",
        "elevated-risk classification",
        "macro_f1",
        True,
        "risk",
        "classification",
    ),
    (
        "exp06_text",
        "Text mental-state screen",
        "7-class text classification",
        "macro_f1",
        True,
        "text",
        "classification",
    ),
    (
        "exp07_prognosis",
        "DASS severity prognosis",
        "multi-output regression",
        "rmse_mean",
        False,
        "prognosis",
        "regression",
    ),
)


def display_name(key: str) -> str:
    return _DISPLAY.get(str(key), str(key).replace("_", " ").title())


def _classification_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparison = data.get("comparison")
    if isinstance(comparison, dict):  # exp03: {model: {val, test}}
        for model, payload in comparison.items():
            metric = (payload or {}).get("test", {})
            rows.append({"key": model, **{k: metric.get(k) for k in _CLASS_METRICS}})
        return rows
    for entry in data.get("ranking", []) or []:  # exp04/06
        if isinstance(entry, dict) and entry.get("model"):
            rows.append({"key": entry["model"], **{k: entry.get(k) for k in _CLASS_METRICS}})
    return rows


_CLASS_METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def _regression_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in data.get("ranking", []) or []:
        if isinstance(entry, dict) and entry.get("model"):
            rows.append(
                {
                    "key": entry["model"],
                    "rmse_mean": entry.get("rmse_mean"),
                    "r2_mean": entry.get("r2_mean"),
                }
            )
    return rows


def comparison_blocks() -> list[dict[str, Any]]:
    """Ranked comparison per experiment, with the model the app serves today."""
    status = inference.availability()
    blocks: list[dict[str, Any]] = []
    for file, title, task, primary, higher, status_key, kind in _SPECS:
        path = repo_path("reports", "metrics", f"{file}.json")
        if not path.exists():
            continue
        data = load_json(path)
        rows = _classification_rows(data) if kind == "classification" else _regression_rows(data)
        rows = [r for r in rows if r.get(primary) is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: r[primary], reverse=higher)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        best = rows[0]["key"]

        info = status.get(status_key, {})
        served_by_groq = info.get("available") and "groq" in str(info.get("version", ""))
        if info.get("available") and not served_by_groq:
            in_use, source = best, f"local artifact · {info.get('version', '')}".strip(" ·")
        elif served_by_groq:
            in_use, source = None, "AI assistant (no local artifact shipped)"
        else:
            in_use, source = None, "not configured"
        for row in rows:
            row["in_use"] = row["key"] == in_use
            row["best"] = row["key"] == best

        blocks.append(
            {
                "title": title,
                "task": task,
                "kind": kind,
                "primary": primary,
                "rows": rows,
                "best": best,
                "in_use": in_use,
                "source": source,
                "served_by_groq": served_by_groq,
            }
        )
    return blocks


def current_model_summary() -> list[dict[str, str]]:
    """One line per experiment: which model answers in the app right now."""
    out: list[dict[str, str]] = []
    for block in comparison_blocks():
        served = display_name(block["in_use"]) if block["in_use"] else "AI assistant"
        out.append(
            {
                "task": block["title"],
                "model": served,
                "detail": block["source"],
            }
        )
    return out
