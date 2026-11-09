"""Slice definitions, confidence intervals, and stage counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from src.analysis.data import (
    GT_COUNT_BINS,
    GT_COUNT_LABELS,
    LOGO_SIZE_BINS,
    LOGO_SIZE_LABELS,
    Row,
)
from src.analysis.metrics import average_precision

SLICES: dict[str, tuple[list[float], list[str], str]] = {
    "logo_area": (LOGO_SIZE_BINS, LOGO_SIZE_LABELS, "logo_area_pct"),
    "gt_count": (GT_COUNT_BINS, GT_COUNT_LABELS, "gt_count"),
    "inlier_ratio": (
        [0.0, 0.1, 0.3, 0.6, 1.0],
        ["0–10%", "10–30%", "30–60%", "60–100%"],
        "inlier_ratio",
    ),
    "coverage": (
        [0.0, 0.1, 0.3, 0.6, 1.0],
        ["0–10%", "10–30%", "30–60%", "60–100%"],
        "coverage",
    ),
    "query_area_mp": (
        [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1000.0],
        ["<0.5", "0.5–1", "1–2", "2–4", "4–8", "≥8"],
        "query_area_mp",
    ),
    "brightness": (
        [0.0, 60.0, 100.0, 140.0, 180.0, 255.0],
        ["<60", "60–100", "100–140", "140–180", "≥180"],
        "query_brightness",
    ),
    "contrast": (
        [0.0, 20.0, 40.0, 60.0, 100.0],
        ["<20", "20–40", "40–60", "≥60"],
        "query_contrast",
    ),
    "blur": (
        [0.0, 50.0, 200.0, 1000.0, 1e12],
        ["<50", "50–200", "200–1000", "≥1000"],
        "query_blur",
    ),
}


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a 95% bootstrap confidence interval for the mean.

    Args:
        values: Numeric values.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        ``(low, high)``; ``(nan, nan)`` for an empty sequence.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_boot, array.size), replace=True)
    means = samples.mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def stage_counts(
    rows: Sequence[Row],
    models: Sequence[str],
) -> dict[str, Counter[str]]:
    """Count failure stages per model.

    Args:
        rows: Enriched rows.
        models: Model names.

    Returns:
        Counters keyed by model.
    """
    counts: dict[str, Counter[str]] = {model: Counter() for model in models}
    for row in rows:
        counts[row.model][row.stage or row.failure] += 1
    return counts


def slice_metrics(
    rows: Sequence[Row],
    models: Sequence[str],
    threshold: float,
    *,
    n_boot: int = 500,
) -> list[dict[str, object]]:
    """Compute hit rate and AP over interpretable slices.

    Args:
        rows: Enriched rows.
        models: Model names.
        threshold: IoU threshold for a true positive.
        n_boot: Bootstrap resamples for confidence intervals.

    Returns:
        One dictionary per model, slice, and bucket.
    """
    result: list[dict[str, object]] = []
    for model in models:
        model_rows = [row for row in rows if row.model == model]
        positives = [row for row in model_rows if not row.negative]
        for name, (bins, labels, field) in SLICES.items():
            values = np.asarray([float(getattr(row, field)) for row in positives])
            for index, label in enumerate(labels):
                lower, upper = bins[index], bins[index + 1]
                mask = (values >= lower) & (
                    (values < upper) if index < len(labels) - 1 else (values <= upper)
                )
                subset = [row for row, keep in zip(positives, mask) if keep]
                if not subset:
                    continue
                hits = [float(row.hit) for row in subset]
                low, high = bootstrap_mean_ci(hits, n_boot=n_boot, seed=index)
                result.append(
                    {
                        "model": model,
                        "slice": name,
                        "bucket": label,
                        "samples": len(subset),
                        "hit_rate": float(np.mean(hits)),
                        "ci_low": low,
                        "ci_high": high,
                        "mean_iou": float(np.mean([row.iou for row in subset])),
                        "ap50": average_precision(subset, threshold),
                    }
                )
    return result
