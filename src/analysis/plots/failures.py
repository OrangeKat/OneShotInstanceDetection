"""Failure-mode plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.analysis.data import FAILURE_LABELS, FAILURE_ORDER, Row
from src.analysis.metrics import average_precision, failure_counts
from src.analysis.plots.common import (
    FAILURE_COLORS,
    model_color,
    plt,
    save_figure,
)


def plot_failure_breakdown(
    rows: Sequence[Row],
    models: Sequence[str],
    path: Path,
) -> None:
    """Plot a stacked failure breakdown per model.

    Args:
        rows: Enriched rows.
        models: Model names.
        path: Output image path.
    """
    counts = failure_counts(rows, models)
    totals = [sum(counts[model].values()) for model in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(models))
    for category in FAILURE_ORDER:
        values = np.asarray(
            [
                counts[model][category] / max(1, totals[index]) * 100.0
                for index, model in enumerate(models)
            ]
        )
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=FAILURE_LABELS[category],
            color=FAILURE_COLORS[category],
            edgecolor="white",
            linewidth=1,
        )
        for bar, value in zip(bars, values):
            if value >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Percent of all pairs")
    ax.set_ylim(0, 100)
    ax.set_title("Failure breakdown")
    ax.legend(
        ncol=3,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    save_figure(fig, path)


def plot_iou_histogram(
    rows: Sequence[Row],
    models: Sequence[str],
    threshold: float,
    path: Path,
) -> None:
    """Plot IoU distributions for positive queries.

    Args:
        rows: Enriched rows.
        models: Model names.
        threshold: Hit threshold.
        path: Output image path.
    """
    positives = [row for row in rows if not row.negative]
    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, model in enumerate(models):
        values = [row.iou for row in positives if row.model == model]
        ax.hist(
            values,
            bins=bins,
            alpha=0.45,
            label=model,
            color=model_color(index, model),
            edgecolor="white",
        )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"hit threshold {threshold:.2f}",
    )
    ax.set_xlabel("IoU")
    ax.set_ylabel("Pairs")
    ax.set_title("IoU distribution (positive queries)")
    ax.legend()
    save_figure(fig, path)


def plot_inliers_vs_iou(
    rows: Sequence[Row],
    models: Sequence[str],
    threshold: float,
    path: Path,
) -> None:
    """Plot inlier count against IoU for each model.

    Args:
        rows: Enriched rows.
        models: Model names.
        threshold: Hit threshold.
        path: Output image path.
    """
    positives = [row for row in rows if not row.negative]
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.6 * len(models), 4.2),
        sharey=True,
    )
    if len(models) == 1:
        axes = [axes]
    for axis, model in zip(axes, models):
        subset = [row for row in positives if row.model == model]
        x_values = np.asarray([row.inliers for row in subset], dtype=float) + 1.0
        y_values = np.asarray([row.iou for row in subset], dtype=float)
        colors = ["#2ca02c" if row.hit else "#d62728" for row in subset]
        axis.scatter(
            x_values,
            y_values,
            c=colors,
            s=10,
            alpha=0.35,
            edgecolors="none",
        )
        axis.set_xscale("log")
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{model} (green=hit, red=miss)")
        axis.set_xlabel("RANSAC inliers + 1 (log)")
    axes[0].set_ylabel("IoU")
    fig.suptitle("Inlier count vs IoU")
    save_figure(fig, path)


def plot_pr_curves(
    rows: Sequence[Row],
    models: Sequence[str],
    threshold: float,
    path: Path,
) -> None:
    """Plot precision-recall curves using inlier count as score.

    Args:
        rows: Enriched rows.
        models: Model names.
        threshold: Hit threshold.
        path: Output image path.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for index, model in enumerate(models):
        model_rows = [row for row in rows if row.model == model]
        positives = sum(1 for row in model_rows if not row.negative)
        detections = [row for row in model_rows if row.predicted]
        if not positives or not detections:
            continue
        detections.sort(key=lambda row: (row.inliers, row.matches), reverse=True)
        tp = 0
        fp = 0
        precisions = [1.0]
        recalls = [0.0]
        for row in detections:
            if not row.negative and row.iou >= threshold:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / positives)
        ap = average_precision(model_rows, threshold)
        ax.plot(
            recalls,
            precisions,
            label=f"{model} AP@0.5={ap:.3f}",
            color=model_color(index, model),
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall (score = inlier count)")
    ax.legend()
    save_figure(fig, path)


def plot_latency(
    rows: Sequence[Row],
    models: Sequence[str],
    path: Path,
) -> None:
    """Plot latency distributions per model.

    Args:
        rows: Enriched rows.
        models: Model names.
        path: Output image path.
    """
    data = [[row.latency_ms for row in rows if row.model == model] for model in models]
    labels = [
        f"{model}\np50={np.percentile(np.asarray(values), 50):.1f} ms\n"
        f"p95={np.percentile(np.asarray(values), 95):.1f} ms"
        for model, values in zip(models, data)
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    boxplot = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    for patch, index, model in zip(boxplot["boxes"], range(len(models)), models):
        patch.set_facecolor(model_color(index, model))
        patch.set_alpha(0.6)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency distribution (outliers hidden)")
    save_figure(fig, path)
