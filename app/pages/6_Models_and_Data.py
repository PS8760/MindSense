"""Models & data: saved metrics, tables, figures and model metadata.

One tab per section plus the ranked dataset table, source tags
(real vs synthetic), model versions and subgroup/fairness tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import model_info, sidebar, theme  # noqa: E402
from mindsense import inference  # noqa: E402
from mindsense.utils.io import load_config, load_json, read_table, repo_path  # noqa: E402

st.set_page_config(page_title="Models & data — MindSense", page_icon="🧪", layout="wide")
theme.apply()
sidebar.render_sidebar()

st.title("🧪 Models & data")
st.caption(
    "Everything the project actually produced — metrics, tables, figures, "
    "model versions. Nothing here is hand-written; files are read from "
    "reports/ and models/."
)

# short human tab label per section; file globs still use expNN names.
EXPERIMENTS = {
    1: (
        "Data pipeline",
        "Acquisition ladder, cleaning, harmonisation, splits, data quality.",
        "Data",
    ),
    2: ("Explore the data", "Distributions, associations, text statistics.", "Explore"),
    3: ("Photo mood model", "CNN on FER-2013 with Grad-CAM (optional module).", "Photos"),
    4: ("Mood screener", "DASS severity bands + a compact short screener.", "Screener"),
    5: ("Clinical terms", "Rule+lexicon extraction of clinical-style mentions.", "Entities"),
    6: ("Risk prediction", "Core model on the harmonised lifestyle data.", "Risk"),
    7: (
        "Text mining",
        "Mental-state classification, Dreaddit external check, drug reviews.",
        "Words",
    ),
    8: ("Explainability", "SHAP/LIME examples and subgroup/fairness tables.", "Explainability"),
}


def _glob_sorted(pattern: str) -> list[Path]:
    return sorted(repo_path("reports").glob(pattern))


def _show_json(path: Path) -> None:
    data = load_json(path)
    st.markdown(f"**{path.name}**")
    if isinstance(data, dict) and all(
        isinstance(v, (int, float, str, bool)) or v is None for v in data.values()
    ):
        st.dataframe(
            pd.DataFrame([data]).T.rename(columns={0: "value"}),
            width="stretch",
        )
    else:
        st.json(data)


def _show_table(path: Path) -> None:
    st.markdown(f"**{path.name}**")
    try:
        df = read_table(path)
    except Exception:
        st.json(load_json(path))
        return
    st.dataframe(df, width="stretch", height=min(420, 38 + 35 * len(df)))


def render_experiment(number: int, title: str, description: str) -> None:
    st.markdown(f"### {title}")
    st.caption(description)
    metrics = _glob_sorted(f"metrics/exp{number:02d}*.json")
    figures = _glob_sorted(f"figures/exp{number:02d}*.png")
    tables = _glob_sorted(f"tables/exp{number:02d}*.csv") + _glob_sorted(
        f"tables/exp{number:02d}*.json"
    )
    if not (metrics or figures or tables):
        st.info(
            f"Nothing produced for **{title}** yet — run its notebook "
            "(`make notebooks`) or `make train` and reload."
        )
        return
    for path in metrics:
        _show_json(path)
    for path in tables:
        _show_table(path)
    if figures:
        st.markdown("**Figures**")
        cols = st.columns(2)
        for i, path in enumerate(figures):
            cols[i % 2].image(path, caption=path.name, width="stretch")


def render_comparison() -> None:
    """Ranked model comparison + the model the app actually serves today."""
    st.markdown("### 🏆 Model comparison")
    st.caption(
        "Each experiment trains several model families, ranks them by a primary "
        "metric on the held-out test split, and the best is served. Numbers are "
        "read from `reports/metrics/` — never hand-written."
    )
    blocks = model_info.comparison_blocks()
    if not blocks:
        st.info("No metrics yet — run `make train`.")
        return
    for block in blocks:
        st.markdown(f"#### {block['title']}")
        st.caption(block["task"])
        rows = block["rows"]
        if block["kind"] == "classification":
            frame = pd.DataFrame(
                [
                    {
                        "Rank": r["rank"],
                        "Model": model_info.display_name(r["key"]),
                        "Accuracy": r["accuracy"],
                        "Precision": r["macro_precision"],
                        "Recall": r["macro_recall"],
                        "F1 (macro)": r["macro_f1"],
                        "Status": (
                            "🏆 best" + (" · ✅ in use" if r["in_use"] else "")
                            if r["best"] or r["in_use"]
                            else ""
                        ),
                    }
                    for r in rows
                ]
            )
        else:
            frame = pd.DataFrame(
                [
                    {
                        "Rank": r["rank"],
                        "Model": model_info.display_name(r["key"]),
                        "RMSE (lower is better)": r["rmse_mean"],
                        "R²": r["r2_mean"],
                        "Status": (
                            "🏆 best" + (" · ✅ in use" if r["in_use"] else "")
                            if r["best"] or r["in_use"]
                            else ""
                        ),
                    }
                    for r in rows
                ]
            )
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                col: st.column_config.NumberColumn(format="%.3f")
                for col in (
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 (macro)",
                    "RMSE (lower is better)",
                    "R²",
                )
            },
        )
        served = model_info.display_name(block["in_use"]) if block["in_use"] else "AI assistant"
        if block["in_use"]:
            st.success(f"✅ Currently used in the app: **{served}** — {block['source']}")
        else:
            st.info(
                f"✅ Currently used in the app: **{served}** — {block['source']}. "
                f"The best trained model (**{model_info.display_name(block['best'])}**) "
                "takes over after `make train`."
            )
        st.write("")


# --------------------------------------------------------------------------- #
# ranked dataset table
# --------------------------------------------------------------------------- #
tab_compare, tab_datasets, *exp_tabs, tab_models = st.tabs(
    ["🏆 Model comparison", "📚 Ranked datasets"]
    + [tab for _n, (_t, _d, tab) in EXPERIMENTS.items()]
    + ["🤖 Models & fairness"]
)

with tab_compare:
    render_comparison()

with tab_datasets:
    st.markdown(
        "Every dataset MindSense is configured to use, merged with what "
        "`scripts/download_data.py` actually found:"
    )
    config = load_config()
    inventory_path = repo_path("reports", "tables", "data_inventory.csv")
    inventory = read_table(inventory_path) if inventory_path.exists() else pd.DataFrame()
    rows = []
    for key, ds in sorted(config["datasets"].items(), key=lambda kv: kv[1]["rank"]):
        row = {
            "rank": ds["rank"],
            "key": key,
            "tier": ds["tier"],
            "population": ds.get("population", ""),
            "licence": ds.get("licence", ""),
            "source": ds.get("source_url", ""),
        }
        if not inventory.empty and key in inventory["key"].values:
            inv = inventory[inventory["key"] == key].iloc[0]
            row.update(
                {
                    "rows": inv.get("rows"),
                    "data_source": inv.get("data_source"),
                    "access_status": inv.get("access_status"),
                    "accessed_on": inv.get("accessed_on"),
                }
            )
        else:
            row.update(
                {
                    "rows": None,
                    "data_source": "not inventoried yet",
                    "access_status": "unknown",
                    "accessed_on": "",
                }
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", height=420)
    st.caption(
        "Tier A = primary, B = supporting, C = optional/external-check. "
        "`data_source` is `real` or `synthetic` — synthetic appears only where "
        "acquisition failed (acquisition ladder)."
    )

for number, (title, description, _tab) in EXPERIMENTS.items():
    with exp_tabs[number - 1]:
        render_experiment(number, title, description)

with tab_models:
    st.markdown("### Registered artifacts")
    status = inference.availability()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "module": info["label"],
                    "available": info["available"],
                    "version": info["version"],
                    "detail": info["detail"],
                }
                for info in status.values()
            ]
        ),
        width="stretch",
    )

    metadata_path = repo_path("models", "metadata.json")
    if metadata_path.exists():
        st.markdown("### models/metadata.json")
        st.json(load_json(metadata_path))
    else:
        st.info("No `models/metadata.json` yet — produced by `make train`.")

    fairness = (
        sorted(repo_path("reports", "tables").glob("*fairness*"))
        if repo_path("reports", "tables").exists()
        else []
    )
    st.markdown("### Subgroup / fairness tables")
    if not fairness:
        st.info(
            "Fairness tables arrive once the fairness report runs (`reports/tables/*fairness*`)."
        )
    for path in fairness:
        _show_table(path)

    st.markdown("### Reproducibility")
    st.markdown(
        "- Global seed: `42` (config.yaml)\n"
        "- Environment: pinned `requirements*.txt` (see `make setup`)\n"
        "- Data split: stratified 70/15/15 as a `split` column inside each parquet\n"
        "- Metrics files are written by the notebooks/training code only — "
        "never edited by hand."
    )

st.divider()
st.caption(json.dumps({"sections": len(EXPERIMENTS), "page": "models_and_data"}, indent=None))
