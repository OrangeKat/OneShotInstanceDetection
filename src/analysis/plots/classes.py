"""Per-class plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.analysis.plots.common import plt, save_figure


def plot_per_class_heatmap(
    class_rows: Sequence[Mapping[str, object]],
    models: Sequence[str],
    metric: str,
    title: str,
    path: Path,
) -> None:
    """Plot a class-by-model heatmap.

    Args:
        class_rows: Per-class metric rows.
        models: Model names.
        metric: Metric key to plot.
        title: Plot title.
        path: Output image path.
    """
    classes = sorted({str(row["class"]) for row in class_rows})
    values = np.full((len(classes), len(models)), np.nan)
    for class_index, class_name in enumerate(classes):
        for model_index, model in enumerate(models):
            matches = [
                row
                for row in class_rows
                if row["class"] == class_name and row["model"] == model
            ]
            if matches:
                values[class_index, model_index] = float(matches[0][metric])
    order = np.argsort(np.nanmean(values, axis=1))
    values = values[order]
    classes = [classes[index] for index in order]

    fig, ax = plt.subplots(figsize=(5.8, 0.31 * len(classes) + 1.8))
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_ylabel("class (worst mean first)")
    ax.set_title(title)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                color = "black" if 0.25 < value < 0.75 else "white"
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=color,
                )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, path)
