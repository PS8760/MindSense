"""Population Insights (Exp 2): interactive Plotly dashboards over the
processed datasets, with filters and honest data-source labelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import sidebar, theme  # noqa: E402
from mindsense.utils.io import repo_path  # noqa: E402

st.set_page_config(page_title="Population Insights — MindSense", page_icon="📊", layout="wide")
theme.apply()
sidebar.render_sidebar()

st.title("📊 Population Insights")
st.caption("Exploratory dashboards over the harmonised datasets (Exp 2) — descriptive only, never causal.")
st.info(
    "Aggregates from the training datasets. Rows tagged **synthetic** appear only "
    "where a real dataset could not be acquired (see the badge below).",
    icon="🗄️",
)

PALETTE = ["#3B7DD8", "#2FA88B", "#E8A13A", "#D95F5F", "#8E6FD8"]


@st.cache_data(show_spinner=False)
def _load(name: str) -> pd.DataFrame:
    path = repo_path("data", "processed", f"{name}.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _source_badge(df: pd.DataFrame) -> None:
    if "data_source" not in df:
        return
    counts = df["data_source"].value_counts()
    if set(counts.index) <= {"real"}:
        st.success("Data source: **100% real** (as acquired).")
    else:
        parts = ", ".join(f"{k}: {v:,}" for k, v in counts.items())
        st.warning(f"Data source mix: {parts} — synthetic rows are labelled, never hidden.")


def _empty(name: str) -> None:
    st.warning(
        f"``data/processed/{name}.parquet`` not found — run ``make data`` first "
        "(Experiment 1)."
    )


dataset = st.radio(
    "Dataset",
    ["Lifestyle risk survey", "DASS-42", "Sentiment corpus"],
    horizontal=True,
)

# --------------------------------------------------------------------------- #
if dataset == "Lifestyle risk survey":
    df = _load("risk_harmonized")
    if df.empty:
        _empty("risk_harmonized")
        st.stop()

    _source_badge(df)
    f1, f2, f3 = st.columns(3)
    populations = ["all", *sorted(df["population"].dropna().unique())]
    population = f1.selectbox("Population", populations)
    genders = ["all", *sorted(df["gender"].dropna().unique())]
    gender = f2.selectbox("Gender", genders)
    age_max = int(df["age"].max())
    age_range = f3.slider("Age range", int(df["age"].min()), age_max, (18, min(35, age_max)))

    view = df.copy()
    if population != "all":
        view = view[view["population"] == population]
    if gender != "all":
        view = view[view["gender"] == gender]
    view = view[(view["age"] >= age_range[0]) & (view["age"] <= age_range[1])]

    if view.empty:
        st.warning("No rows match these filters.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows in view", f"{len(view):,}")
    k2.metric("Elevated-risk rate", f"{view['mh_risk'].mean():.1%}")
    k3.metric("Median age", f"{view['age'].median():.0f}")
    k4.metric("Median sleep", f"{view['sleep_hours'].median():.1f} h")

    view = view.assign(
        age_band=pd.cut(
            view["age"],
            bins=[9, 19, 24, 34, 49, 200],
            labels=["≤19", "20–24", "25–34", "35–49", "50+"],
        )
    )
    rate = (
        view.groupby(["age_band", "population"], observed=True)["mh_risk"]
        .mean()
        .reset_index()
    )
    fig1 = px.bar(
        rate, x="age_band", y="mh_risk", color="population", barmode="group",
        color_discrete_sequence=PALETTE,
        title="Elevated-risk rate by age band and population",
        labels={"mh_risk": "Share flagged", "age_band": "Age band"},
    )
    fig1.update_layout(template="plotly_white", yaxis_tickformat=".0%", height=380)

    fig2 = px.box(
        view.dropna(subset=["stress_pressure_score"]),
        x="population", y="stress_pressure_score", color="population",
        color_discrete_sequence=PALETTE,
        title="Stress / pressure score distribution (0–5)",
        points=False,
    )
    fig2.update_layout(template="plotly_white", height=380, showlegend=False)

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(fig1, width="stretch")
    col_b.plotly_chart(fig2, width="stretch")

    sleep_view = view.dropna(subset=["sleep_hours"])
    if not sleep_view.empty:
        fig3 = px.histogram(
            sleep_view, x="sleep_hours", color="mh_risk", barmode="overlay",
            nbins=18, color_discrete_sequence=["#2FA88B", "#D95F5F"],
            title="Sleep hours by risk label (0 = no concern, 1 = flagged)",
            opacity=0.65,
        )
        fig3.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig3, width="stretch")

    st.caption(
        "Descriptive statistics of the harmonised survey (student + workplace "
        "populations). Differences between groups are associations in the data — "
        "they are not causes and not clinical facts about any individual."
    )

# --------------------------------------------------------------------------- #
elif dataset == "DASS-42":
    df = _load("dass42")
    if df.empty:
        _empty("dass42")
        st.stop()

    _source_badge(df)
    bands = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]
    f1, f2 = st.columns(2)
    gender_filter = f1.selectbox("Gender", ["all", *sorted(df["gender"].dropna().unique())])
    band_filter = f2.multiselect("Depression band", bands, default=bands)

    view = df.copy()
    if gender_filter != "all":
        view = view[view["gender"] == gender_filter]
    view = view[view["depression_band"].isin(band_filter)]

    if view.empty:
        st.warning("No rows match these filters.")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Rows in view", f"{len(view):,}")
    k2.metric("Median depression score", f"{view['depression_score'].median():.0f}")
    k3.metric("Median anxiety score", f"{view['anxiety_score'].median():.0f}")

    band_counts = (
        view["depression_band"].value_counts().reindex(bands).dropna().reset_index()
    )
    band_counts.columns = ["band", "count"]
    fig1 = px.bar(
        band_counts, x="band", y="count", color="band",
        color_discrete_sequence=PALETTE, title="DASS depression bands",
        category_orders={"band": bands},
    )
    fig1.update_layout(template="plotly_white", height=380, showlegend=False)

    fig2 = px.histogram(
        view, x="depression_score", nbins=30, color_discrete_sequence=["#3B7DD8"],
        title="Depression subscale score (0–42)",
    )
    fig2.update_layout(template="plotly_white", height=380, showlegend=False)

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(fig1, width="stretch")
    col_b.plotly_chart(fig2, width="stretch")

    fig3 = px.histogram(
        view, x="depression_score", color="gender", barmode="overlay",
        nbins=25, color_discrete_sequence=PALETTE,
        title="Depression score by gender (overlaid)", opacity=0.6,
    )
    fig3.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig3, width="stretch")

    st.caption(
        "DASS-42 online sample (openpsychometrics.org) — a self-selected web "
        "sample, not a population prevalence study. Instrument bands applied to "
        "raw subscale sums (see DECISIONS for the scoring rationale)."
    )

# --------------------------------------------------------------------------- #
else:
    df = _load("sentiment_mh")
    if df.empty:
        _empty("sentiment_mh")
        st.stop()

    _source_badge(df)
    label_order = list(df["label"].value_counts().index)
    f1, f2 = st.columns(2)
    label_filter = f1.multiselect("Labels", label_order, default=label_order)
    min_len, max_len = 0, int(df["text_norm"].str.len().quantile(0.99))
    len_range = f2.slider("Text length (chars)", 0, max(int(df['text_norm'].str.len().max()), 1),
                          (0, min(600, max_len)))

    view = df[df["label"].isin(label_filter)]
    view = view[
        (view["text_norm"].str.len() >= len_range[0])
        & (view["text_norm"].str.len() <= len_range[1])
    ]
    if view.empty:
        st.warning("No rows match these filters.")
        st.stop()

    k1, k2 = st.columns(2)
    k1.metric("Rows in view", f"{len(view):,}")
    k2.metric("Classes", f"{view['label'].nunique()}")

    counts = view["label"].value_counts().reindex(label_order).dropna().reset_index()
    counts.columns = ["label", "count"]
    fig1 = px.bar(
        counts, x="label", y="count", color="label",
        color_discrete_sequence=PALETTE, title="Mental-state label distribution",
    )
    fig1.update_layout(template="plotly_white", height=380, showlegend=False)

    fig2 = px.box(
        view, x="label", y=view["text_norm"].str.len().clip(upper=max_len),
        color="label", color_discrete_sequence=PALETTE, points=False,
        title=f"Text length by label (clipped at p99 = {max_len} chars)",
    )
    fig2.update_layout(template="plotly_white", height=380, showlegend=False,
                       yaxis_title="length (chars)")
    fig2.update_xaxes(categoryorder="array", categoryarray=label_order)

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(fig1, width="stretch")
    col_b.plotly_chart(fig2, width="stretch")

    sample = view.sample(min(6, len(view)), random_state=42)
    st.markdown("**Sample statements (anonymised dataset rows):**")
    for _, row in sample.iterrows():
        st.markdown(f"- **{row['label']}** — {row['text'][:180]}")

    st.caption(
        "Labelled mental-health statements (Kaggle sentiment corpus). Labels are "
        "dataset annotations, not diagnoses of real individuals; the corpus "
        "contains some Dreaddit overlap documented in Lab Results."
    )

st.divider()
st.caption("Charts: Plotly · filters affect this page only · seed 42 for any sampling.")
