"""Markdown diagnostics for stages, slices, and model efficiency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from src.analysis.slices import SLICES

STAGE_ORDER = [
    "hit",
    "wrong_box",
    "no_matches",
    "too_few_matches",
    "ransac_failed",
    "degenerate_homography",
    "no_prediction",
    "false_positive",
    "true_negative",
]
REPORT_SLICES = ["logo_area", "gt_count", "inlier_ratio", "coverage", "query_area_mp"]


def render_diagnostics(
    payload: Mapping[str, object],
    stages: Mapping[str, Counter[str]],
    slices: Sequence[Mapping[str, object]],
    models: Sequence[str],
    threshold: float,
) -> str:
    """Render stage, slice, and efficiency diagnostics.

    Args:
        payload: Benchmark results payload.
        stages: Stage counters per model.
        slices: Slice metric rows.
        models: Model names.
        threshold: IoU threshold for a hit.

    Returns:
        Markdown document.
    """
    reports = {str(report["model"]): report for report in payload["models"]}
    lines = [
        "# Diagnostics",
        "",
        f"- Hit threshold: `IoU >= {threshold:.2f}`",
        "- Stage labels come from the per-sample record; confidence intervals "
        "are 95% bootstrap intervals over pairs.",
        "",
        "## Failure stages",
        "",
    ]
    present = [stage for stage in STAGE_ORDER if any(stages[m][stage] for m in models)]
    lines.append("| Model | " + " | ".join(present) + " |")
    lines.append("|---|" + "---:|" * len(present))
    for model in models:
        total = max(1, sum(stages[model].values()))
        cells = [
            f"{stages[model][stage]} ({stages[model][stage] / total * 100:.1f}%)"
            for stage in present
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Efficiency and matching",
            "",
            "| Model | Params (M) | Valid H | Inlier ratio | Coverage "
            "| Median reproj (px) | p50 ms | p95 ms | Peak GPU (MiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        report = reports[model]
        latency = report["latency_ms"]
        params = report.get("params")
        lines.append(
            f"| {model} "
            f"| {_m(params)} "
            f"| {float(report.get('valid_homography_rate', 0.0)):.3f} "
            f"| {float(report.get('mean_inlier_ratio', 0.0)):.3f} "
            f"| {float(report.get('mean_coverage', 0.0)):.3f} "
            f"| {_px(report.get('median_reprojection_error'))} "
            f"| {float(latency['p50']):.1f} "
            f"| {float(latency['p95']):.1f} "
            f"| {float(report.get('peak_gpu_mib', 0.0)):.0f} |"
        )

    for name in REPORT_SLICES:
        if name not in SLICES:
            continue
        lines.extend(_slice_table(name, slices, models))
    return "\n".join(lines)


def _slice_table(
    name: str,
    slices: Sequence[Mapping[str, object]],
    models: Sequence[str],
) -> list[str]:
    """Render one slice table.

    Args:
        name: Slice name.
        slices: Slice metric rows.
        models: Model names.

    Returns:
        Markdown lines.
    """
    _, labels, _ = SLICES[name]
    subset = [row for row in slices if row["slice"] == name]
    if not subset:
        return []
    buckets = [
        label for label in labels if any(row["bucket"] == label for row in subset)
    ]
    lines = [
        "",
        f"## Slice: {name.replace('_', ' ')}",
        "",
        "| Bucket | Pairs | " + " | ".join(models) + " |",
        "|---|---:|" + "---:|" * len(models),
    ]
    for bucket in buckets:
        cells = []
        pair_count = 0
        for model in models:
            row = next(
                (
                    item
                    for item in subset
                    if item["model"] == model and item["bucket"] == bucket
                ),
                None,
            )
            if row is None:
                cells.append("n/a")
                continue
            pair_count = max(pair_count, int(row["samples"]))
            cells.append(
                f"{float(row['hit_rate']):.3f} "
                f"[{float(row['ci_low']):.3f}–{float(row['ci_high']):.3f}] "
                f"(n={int(row['samples'])})"
            )
        lines.append(f"| {bucket} | {pair_count} | " + " | ".join(cells) + " |")
    return lines


def _m(value: object) -> str:
    """Format a parameter count in millions.

    Args:
        value: Parameter count or ``None``.

    Returns:
        Formatted value.
    """
    if value is None:
        return "n/a"
    return f"{float(value) / 1e6:.1f}"


def _px(value: object) -> str:
    """Format an optional pixel value.

    Args:
        value: Pixel value or ``None``.

    Returns:
        Formatted value.
    """
    return "n/a" if value is None else f"{float(value):.2f}"
