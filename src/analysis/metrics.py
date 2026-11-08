"""Metric aggregation for benchmark failure analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from src.analysis.data import Row


def average_precision(rows: Sequence[Row], threshold: float) -> float:
    """Compute 11-point AP from per-pair records.

    Args:
        rows: Rows for one model and class.
        threshold: IoU threshold for a true positive.

    Returns:
        Average precision.
    """
    positives = sum(1 for row in rows if not row.negative)
    if positives == 0:
        return 0.0
    detections = [row for row in rows if row.predicted]
    detections.sort(key=lambda row: (row.inliers, row.matches), reverse=True)
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for row in detections:
        if not row.negative and row.iou >= threshold:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / positives)
    total = 0.0
    for point in np.linspace(0.0, 1.0, 11):
        total += max(
            (
                precision
                for precision, recall in zip(precisions, recalls)
                if recall >= point
            ),
            default=0.0,
        )
    return total / 11.0


def per_class_metrics(
    rows: Sequence[Row],
    models: Sequence[str],
    threshold: float,
) -> list[dict[str, object]]:
    """Compute per-class metrics for each model.

    Args:
        rows: Enriched rows.
        models: Model names.
        threshold: IoU threshold for a true positive.

    Returns:
        Per-class metric dictionaries.
    """
    result: list[dict[str, object]] = []
    for model in models:
        model_rows = [row for row in rows if row.model == model]
        by_class: dict[str, list[Row]] = defaultdict(list)
        for row in model_rows:
            by_class[row.class_name].append(row)
        for class_name in sorted(by_class):
            class_rows = by_class[class_name]
            positives = [row for row in class_rows if not row.negative]
            if not positives:
                continue
            hits = sum(1 for row in positives if row.hit)
            result.append(
                {
                    "model": model,
                    "class": class_name,
                    "samples": len(class_rows),
                    "positives": len(positives),
                    "hit_rate": hits / len(positives),
                    "ap50": average_precision(class_rows, threshold),
                    "mean_iou": float(np.mean([row.iou for row in positives])),
                    "mean_matches": float(np.mean([row.matches for row in class_rows])),
                    "mean_inliers": float(np.mean([row.inliers for row in class_rows])),
                    "mean_latency_ms": float(
                        np.mean([row.latency_ms for row in class_rows])
                    ),
                }
            )
    return result


def failure_counts(
    rows: Sequence[Row], models: Sequence[str]
) -> dict[str, Counter[str]]:
    """Count failure categories per model.

    Args:
        rows: Enriched rows.
        models: Model names.

    Returns:
        Counters keyed by model.
    """
    counts: dict[str, Counter[str]] = {model: Counter() for model in models}
    for row in rows:
        counts[row.model][row.failure] += 1
    return counts


def binned_hit_rates(
    rows: Sequence[Row],
    models: Sequence[str],
    field: str,
    bins: Sequence[float],
    labels: Sequence[str],
) -> dict[str, tuple[list[int], list[float]]]:
    """Compute hit rates over numeric bins for positive samples.

    Args:
        rows: Enriched rows.
        models: Model names.
        field: Row attribute to bin.
        bins: Bin edges.
        labels: Bin labels.

    Returns:
        Per-model ``(counts, hit_rates)``.
    """
    result: dict[str, tuple[list[int], list[float]]] = {}
    for model in models:
        positives = [row for row in rows if row.model == model and not row.negative]
        values = np.asarray([float(getattr(row, field)) for row in positives])
        hits = np.asarray([row.hit for row in positives])
        counts: list[int] = []
        rates: list[float] = []
        for index in range(len(labels)):
            lower = bins[index]
            upper = bins[index + 1]
            mask = (values >= lower) & (
                (values < upper) if index < len(labels) - 1 else (values <= upper)
            )
            count = int(mask.sum())
            counts.append(count)
            rates.append(float(hits[mask].mean()) if count else float("nan"))
        result[model] = (counts, rates)
    return result


def hit_rate(rows: Sequence[Row]) -> float:
    """Return the hit rate for a set of rows.

    Args:
        rows: Rows to summarize.

    Returns:
        Hit rate, or ``nan`` for an empty set.
    """
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row.hit) / len(rows)
