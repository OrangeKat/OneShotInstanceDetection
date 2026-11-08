"""Markdown failure-analysis report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.analysis.data import (
    FAILURE_ORDER,
    GT_COUNT_BINS,
    GT_COUNT_LABELS,
    LOGO_SIZE_BINS,
    LOGO_SIZE_LABELS,
    Row,
)
from src.analysis.metrics import binned_hit_rates, failure_counts
from src.analysis.tables import _class_table, _observations


def render_report(
    rows: Sequence[Row],
    payload: Mapping[str, object],
    class_rows: Sequence[Mapping[str, object]],
    models: Sequence[str],
    threshold: float,
    top_classes: int,
    results_path: Path,
    manifest_path: Path,
) -> str:
    """Render the Markdown failure-analysis report.

    Args:
        rows: Enriched rows.
        payload: Results payload.
        class_rows: Per-class metric rows.
        models: Model names.
        threshold: Hit threshold.
        top_classes: Number of best/worst classes per model.
        results_path: Results JSON path.
        manifest_path: Manifest path.

    Returns:
        Markdown document.
    """
    reports = {str(report["model"]): report for report in payload["models"]}
    counts = failure_counts(rows, models)
    lines = [
        "# Failure analysis",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Results: `{results_path}`",
        f"- Hit threshold: `IoU >= {threshold:.2f}`",
        "",
        "## Model summary",
        "",
        "| Model | Samples | Positives | Hit rate | AP@0.5 | Mean IoU "
        "| Mean matches | Mean inliers | p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        report = reports[model]
        latency = report["latency_ms"]
        lines.append(
            f"| {model} | {report['samples']} | {report['positives']} "
            f"| {float(report['hit_rate']):.3f} | {float(report['ap50']):.3f} "
            f"| {float(report['mean_iou']):.3f} "
            f"| {float(report['mean_matches']):.1f} "
            f"| {float(report['mean_inliers']):.1f} "
            f"| {float(latency['p50']):.1f} ms "
            f"| {float(latency['p95']):.1f} ms |"
        )

    lines.extend(
        [
            "",
            "## Failure breakdown",
            "",
            "| Model | Hit | Wrong box | No prediction | False positive "
            "| True negative |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        total = max(1, sum(counts[model].values()))
        cells = [
            f"{counts[model][category]} ({counts[model][category] / total * 100:.1f}%)"
            for category in FAILURE_ORDER
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Worst classes", ""])
    lines.extend(_class_table(class_rows, models, top_classes, worst=True))
    lines.extend(["", "## Best classes", ""])
    lines.extend(_class_table(class_rows, models, top_classes, worst=False))

    size_data = binned_hit_rates(
        rows,
        models,
        "logo_area_pct",
        LOGO_SIZE_BINS,
        LOGO_SIZE_LABELS,
    )
    lines.extend(
        [
            "",
            "## Hit rate by logo size",
            "",
            "| Largest GT box / query | Pairs | " + " | ".join(models) + " |",
            "|---|---:|" + "---:|" * len(models),
        ]
    )
    shared_counts = size_data[models[0]][0] if models else []
    for index, label in enumerate(LOGO_SIZE_LABELS):
        cells = []
        for model in models:
            rate = size_data[model][1][index]
            cells.append("n/a" if np.isnan(rate) else f"{rate:.3f}")
        lines.append(
            f"| {label} | {shared_counts[index]} | " + " | ".join(cells) + " |"
        )

    count_data = binned_hit_rates(
        rows,
        models,
        "gt_count",
        GT_COUNT_BINS,
        GT_COUNT_LABELS,
    )
    lines.extend(
        [
            "",
            "## Hit rate by GT instance count",
            "",
            "| GT instances | Pairs | " + " | ".join(models) + " |",
            "|---|---:|" + "---:|" * len(models),
        ]
    )
    shared_counts = count_data[models[0]][0] if models else []
    for index, label in enumerate(GT_COUNT_LABELS):
        cells = []
        for model in models:
            rate = count_data[model][1][index]
            cells.append("n/a" if np.isnan(rate) else f"{rate:.3f}")
        lines.append(
            f"| {label} | {shared_counts[index]} | " + " | ".join(cells) + " |"
        )

    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- [Failure breakdown](failure_breakdown.png)",
            "- [Per-class hit rate](per_class_hit_rate.png)",
            "- [Per-class mean IoU](per_class_mean_iou.png)",
            "- [IoU distribution](iou_distribution.png)",
            "- [Hit rate by logo size](hit_rate_by_logo_size.png)",
            "- [Hit rate by GT instance count](hit_rate_by_gt_count.png)",
            "- [Inliers vs IoU](inliers_vs_iou.png)",
            "- [Precision-recall](precision_recall.png)",
            "- [Latency distribution](latency_distribution.png)",
            "",
            "## Observations",
            "",
        ]
    )
    lines.extend(_observations(rows, reports, class_rows, models, threshold))
    lines.append("")
    return "\n".join(lines)
