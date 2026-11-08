"""Hit-rate factor plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.analysis.data import (
    GT_COUNT_BINS,
    GT_COUNT_LABELS,
    LOGO_SIZE_BINS,
    LOGO_SIZE_LABELS,
    Row,
)
from src.analysis.metrics import binned_hit_rates
from src.analysis.plots.common import model_color, plt, save_figure


def plot_hit_rate_by_logo_size(
    rows: Sequence[Row],
    models: Sequence[str],
    path: Path,
) -> None:
    """Plot hit rate by logo size as a share of the query image.

    Args:
        rows: Enriched rows.
        models: Model names.
        path: Output image path.
    """
    data = binned_hit_rates(
        rows,
        models,
        "logo_area_pct",
        LOGO_SIZE_BINS,
        LOGO_SIZE_LABELS,
    )
    fig, (ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = np.arange(len(LOGO_SIZE_LABELS))
    width = 0.8 / len(models)
    for index, model in enumerate(models):
        counts, rates = data[model]
        offsets = x - 0.4 + width * (index + 0.5)
        values = [0.0 if np.isnan(rate) else rate for rate in rates]
        ax.bar(
            offsets,
            values,
            width=width,
            label=model,
            color=model_color(index, model),
        )
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1)
    ax.set_title("Hit rate by logo size (largest GT box / query image)")
    ax.legend(fontsize=8)
    shared_counts = data[models[0]][0] if models else []
    count_ax.bar(x, shared_counts, color="#7f7f7f", width=0.6)
    count_ax.set_ylabel("Pairs")
    count_ax.set_xticks(x)
    count_ax.set_xticklabels(LOGO_SIZE_LABELS)
    count_ax.set_xlabel("Largest GT logo box as % of query image")
    save_figure(fig, path)


def plot_hit_rate_by_gt_count(
    rows: Sequence[Row],
    models: Sequence[str],
    path: Path,
) -> None:
    """Plot hit rate by the number of ground-truth instances.

    Args:
        rows: Enriched rows.
        models: Model names.
        path: Output image path.
    """
    data = binned_hit_rates(
        rows,
        models,
        "gt_count",
        GT_COUNT_BINS,
        GT_COUNT_LABELS,
    )
    fig, (ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(8, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = np.arange(len(GT_COUNT_LABELS))
    width = 0.8 / len(models)
    for index, model in enumerate(models):
        counts, rates = data[model]
        offsets = x - 0.4 + width * (index + 0.5)
        values = [0.0 if np.isnan(rate) else rate for rate in rates]
        ax.bar(
            offsets,
            values,
            width=width,
            label=model,
            color=model_color(index, model),
        )
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1)
    ax.set_title("Hit rate by number of GT instances in the query")
    ax.legend(fontsize=8)
    shared_counts = data[models[0]][0] if models else []
    count_ax.bar(x, shared_counts, color="#7f7f7f", width=0.6)
    count_ax.set_ylabel("Pairs")
    count_ax.set_xticks(x)
    count_ax.set_xticklabels(GT_COUNT_LABELS)
    count_ax.set_xlabel("GT instances")
    save_figure(fig, path)
