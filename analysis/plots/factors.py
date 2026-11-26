"""Hit-rate factor plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from analysis.data import (
    GT_COUNT_BINS,
    GT_COUNT_LABELS,
    OBJECT_SIZE_BINS,
    OBJECT_SIZE_LABELS,
    Row,
)
from analysis.metrics import binned_hit_rates
from analysis.plots.common import model_color, plt, save_figure


def plot_hit_rate_by_object_size(
    rows: Sequence[Row],
    models: Sequence[str],
    path: Path,
) -> None:
    """Plot hit rate by object size as a share of the query image.

    Args:
        rows: Enriched rows.
        models: Model names.
        path: Output image path.
    """
    data = binned_hit_rates(
        rows,
        models,
        "object_area_pct",
        OBJECT_SIZE_BINS,
        OBJECT_SIZE_LABELS,
    )
    fig, (ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = np.arange(len(OBJECT_SIZE_LABELS))
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
    ax.set_title("Hit rate by object size (largest GT box / query image)")
    ax.legend(fontsize=8)
    shared_counts = data[models[0]][0] if models else []
    count_ax.bar(x, shared_counts, color="#7f7f7f", width=0.6)
    count_ax.set_ylabel("Pairs")
    count_ax.set_xticks(x)
    count_ax.set_xticklabels(OBJECT_SIZE_LABELS)
    count_ax.set_xlabel("Largest GT box as % of query image")
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


def plot_agreement_counts(
    distribution: Sequence[int],
    path: Path,
) -> None:
    """Plot how many pairs each number of models solves.

    Args:
        distribution: Counts indexed by the number of successful models.
        path: Output path.
    """
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(distribution))
    colors = ["#d62728", *(["#1f77b4"] * (len(distribution) - 1))]
    ax.bar(x, distribution, color=colors, width=0.65)
    ax.set_xlabel("Models that hit the pair")
    ax.set_ylabel("Positive pairs")
    ax.set_title("Model agreement on positives (0 = no model solves the pair)")
    ax.set_xticks(x)
    for index, count in enumerate(distribution):
        ax.annotate(
            str(count),
            (index, count),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(fig, path)
