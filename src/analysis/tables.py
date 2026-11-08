"""CSV writers and Markdown table helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.analysis.data import FAILURE_LABELS, FAILURE_ORDER, Row
from src.analysis.metrics import failure_counts, hit_rate


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write metric rows as CSV.

    Args:
        path: Output path.
        rows: Rows to write.
    """
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_csv(
    path: Path,
    rows: Sequence[Row],
    models: Sequence[str],
) -> None:
    """Write failure counts as CSV.

    Args:
        path: Output path.
        rows: Enriched rows.
        models: Model names.
    """
    counts = failure_counts(rows, models)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *FAILURE_ORDER])
        for model in models:
            writer.writerow(
                [model, *(counts[model][category] for category in FAILURE_ORDER)]
            )


def _class_table(
    class_rows: Sequence[Mapping[str, object]],
    models: Sequence[str],
    top_classes: int,
    *,
    worst: bool,
) -> list[str]:
    """Render a best/worst class table.

    Args:
        class_rows: Per-class metric rows.
        models: Model names.
        top_classes: Number of classes per model.
        worst: Whether to sort ascending or descending.

    Returns:
        Markdown lines.
    """
    lines = [
        "| Model | Class | Hit rate | AP@0.5 | Mean IoU | Positives |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        subset = [row for row in class_rows if row["model"] == model]
        subset.sort(
            key=lambda row: (float(row["hit_rate"]), float(row["ap50"])),
            reverse=not worst,
        )
        for row in subset[:top_classes]:
            lines.append(
                f"| {model} | {row['class']} | {float(row['hit_rate']):.3f} "
                f"| {float(row['ap50']):.3f} | {float(row['mean_iou']):.3f} "
                f"| {row['positives']} |"
            )
    return lines


def _observations(
    rows: Sequence[Row],
    reports: Mapping[str, Mapping[str, object]],
    class_rows: Sequence[Mapping[str, object]],
    models: Sequence[str],
    threshold: float,
) -> list[str]:
    """Generate short data-driven observations.

    Args:
        rows: Enriched rows.
        reports: Results reports by model.
        class_rows: Per-class metric rows.
        models: Model names.
        threshold: Hit threshold.

    Returns:
        Markdown bullet lines.
    """
    if not models:
        return ["- No models were evaluated."]
    best_ap = max(models, key=lambda model: float(reports[model]["ap50"]))
    fastest = min(
        models,
        key=lambda model: float(reports[model]["latency_ms"]["p50"]),
    )
    counts = failure_counts(rows, models)
    failure_categories = ["wrong_box", "no_prediction", "false_positive"]
    lines = [
        f"- Best AP: **{best_ap}** ({float(reports[best_ap]['ap50']):.3f}).",
        f"- Fastest p50: **{fastest}** "
        f"({float(reports[fastest]['latency_ms']['p50']):.1f} ms).",
    ]
    for model in models:
        dominant = max(
            failure_categories,
            key=lambda category: counts[model][category],
        )
        worst_class = min(
            (row for row in class_rows if row["model"] == model),
            key=lambda row: (float(row["hit_rate"]), float(row["ap50"])),
            default=None,
        )
        worst_text = (
            f"; worst class: **{worst_class['class']}** "
            f"({float(worst_class['hit_rate']):.3f} hit)"
            if worst_class is not None
            else ""
        )
        lines.append(
            f"- {model}: dominant failure is "
            f"**{FAILURE_LABELS[dominant].lower()}** "
            f"({counts[model][dominant]} pairs){worst_text}."
        )

    for model in models:
        positives = [row for row in rows if row.model == model and not row.negative]
        small = [row for row in positives if row.logo_area_pct < 1.0]
        large = [row for row in positives if row.logo_area_pct >= 5.0]
        single = [row for row in positives if row.gt_count == 1]
        multi = [row for row in positives if row.gt_count >= 2]
        small_rate = hit_rate(small)
        large_rate = hit_rate(large)
        single_rate = hit_rate(single)
        multi_rate = hit_rate(multi)
        inliers = np.asarray([row.inliers for row in positives], dtype=float)
        ious = np.asarray([row.iou for row in positives], dtype=float)
        correlation = (
            float(np.corrcoef(inliers, ious)[0, 1])
            if len(inliers) > 2 and inliers.std() > 0
            else float("nan")
        )
        correlation_text = f"{correlation:.2f}" if np.isfinite(correlation) else "n/a"
        lines.append(
            f"- {model} size effect: logos <1% of the query hit "
            f"**{small_rate:.3f}** ({len(small)} pairs) vs "
            f"**{large_rate:.3f}** for >=5% ({len(large)} pairs); "
            f"single-instance {single_rate:.3f} vs multi-instance "
            f"{multi_rate:.3f}; inlier/IoU correlation {correlation_text}."
        )
    return lines
