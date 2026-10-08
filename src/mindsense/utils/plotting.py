"""Plotting helpers shared by notebooks and the app.

Matplotlib figures are saved to ``reports/figures`` with a standard style;
Plotly figures returned to the Streamlit app need no disk IO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe for notebooks/CI
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from mindsense.utils.io import ensure_dir  # noqa: E402

PALETTE = ["#3B7DD8", "#2FA88B", "#E8A13A", "#D95F5F", "#8E6FD8", "#7A8B99", "#C97B2B"]
FIGSIZE = (9, 5.2)
DPI = 140


def style_axes(ax: Any, title: str, xlabel: str = "", ylabel: str = "Count") -> Any:
    """Apply the house style to a matplotlib axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    return ax


def save_fig(fig: Any, name: str, *, subdir: str = "figures") -> Path:
    """Save a figure under ``reports/<subdir>/name.png`` and return the path."""
    from mindsense.utils.io import repo_path

    out = repo_path("reports", subdir, f"{name}.png")
    ensure_dir(out.parent)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def barplot(
    data: Any,
    x: str,
    y: str | None = None,
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "Count",
    hue: str | None = None,
    figsize: tuple[float, float] = FIGSIZE,
) -> Any:
    """Seaborn barplot with house style (returns the figure)."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, palette=PALETTE, edgecolor=None)
    style_axes(ax, title, xlabel, ylabel)
    if hue:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
