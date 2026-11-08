"""Plot functions for benchmark analysis."""

from src.analysis.plots.classes import plot_per_class_heatmap
from src.analysis.plots.factors import (
    plot_hit_rate_by_gt_count,
    plot_hit_rate_by_logo_size,
)
from src.analysis.plots.failures import (
    plot_failure_breakdown,
    plot_inliers_vs_iou,
    plot_iou_histogram,
    plot_latency,
    plot_pr_curves,
)

__all__ = [
    "plot_failure_breakdown",
    "plot_hit_rate_by_gt_count",
    "plot_hit_rate_by_logo_size",
    "plot_inliers_vs_iou",
    "plot_iou_histogram",
    "plot_latency",
    "plot_per_class_heatmap",
    "plot_pr_curves",
]
