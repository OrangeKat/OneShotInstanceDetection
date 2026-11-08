#!/usr/bin/env python3
"""Analyze benchmark results and visualize failure modes.

Reads the per-sample JSON written by ``src/pipeline.py`` and produces
per-class heatmaps, failure breakdowns, hit-rate factor plots, PR curves,
latency plots, a Markdown report, and per-class CSV.

Usage:
    uv run python scripts/analyze_results.py
    uv run python scripts/analyze_results.py --top-classes 8
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.data import load_rows
from src.analysis.metrics import per_class_metrics
from src.analysis.plots import (
    plot_failure_breakdown,
    plot_hit_rate_by_gt_count,
    plot_hit_rate_by_logo_size,
    plot_inliers_vs_iou,
    plot_iou_histogram,
    plot_latency,
    plot_per_class_heatmap,
    plot_pr_curves,
)
from src.analysis.report import render_report
from src.analysis.tables import write_csv, write_failure_csv


def main(argv: Sequence[str] | None = None) -> int:
    """Run result analysis and write plots plus a report.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Analyze prediction failures.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
        help="Benchmark manifest.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/flickrlogos32/results.json"),
        help="Results JSON from src/pipeline.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/flickrlogos32/analysis"),
        help="Directory for plots, CSV, and report.",
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=5,
        help="Number of best/worst classes per model.",
    )
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.results.read_text())
    rows, config = load_rows(args.results, args.manifest)
    models = [str(report["model"]) for report in payload["models"]]
    threshold = float(config["iou_threshold"])

    class_rows = per_class_metrics(rows, models, threshold)
    write_csv(args.output / "per_class_metrics.csv", class_rows)
    write_failure_csv(args.output / "failure_breakdown.csv", rows, models)

    plot_failure_breakdown(rows, models, args.output / "failure_breakdown.png")
    plot_per_class_heatmap(
        class_rows,
        models,
        "hit_rate",
        "Per-class hit rate",
        args.output / "per_class_hit_rate.png",
    )
    plot_per_class_heatmap(
        class_rows,
        models,
        "mean_iou",
        "Per-class mean IoU",
        args.output / "per_class_mean_iou.png",
    )
    plot_iou_histogram(rows, models, threshold, args.output / "iou_distribution.png")
    plot_hit_rate_by_logo_size(rows, models, args.output / "hit_rate_by_logo_size.png")
    plot_hit_rate_by_gt_count(rows, models, args.output / "hit_rate_by_gt_count.png")
    plot_inliers_vs_iou(rows, models, threshold, args.output / "inliers_vs_iou.png")
    plot_pr_curves(rows, models, threshold, args.output / "precision_recall.png")
    plot_latency(rows, models, args.output / "latency_distribution.png")

    report = render_report(
        rows,
        payload,
        class_rows,
        models,
        threshold,
        args.top_classes,
        args.results,
        args.manifest,
    )
    (args.output / "analysis.md").write_text(report)
    print(f"wrote analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
