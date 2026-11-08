"""Shared plotting helpers and colors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "sift": "#1f77b4",
    "orb": "#ff7f0e",
    "akaze": "#2ca02c",
}
FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
FAILURE_COLORS = {
    "hit": "#2ca02c",
    "wrong_box": "#ff7f0e",
    "no_prediction": "#d62728",
    "false_positive": "#9467bd",
    "true_negative": "#bdbdbd",
}


def model_color(index: int, model: str) -> str:
    """Return a stable color for a model.

    Args:
        index: Model index.
        model: Model name.

    Returns:
        Hex color.
    """
    return COLORS.get(model, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close a figure.

    Args:
        fig: Matplotlib figure.
        path: Output path.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
