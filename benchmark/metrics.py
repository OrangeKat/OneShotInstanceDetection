"""Hit rate, mean IoU, and average precision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from benchmark.serialization import record_dict
from benchmark.types import SampleResult


def average_precision(records: Sequence[SampleResult], threshold: float) -> float:
    """Compute 11-point average precision at an IoU threshold.

    Args:
        records: Per-sample records.
        threshold: IoU threshold that defines a true positive.

    Returns:
        The average precision.
    """
    positives = sum(1 for record in records if record.gt_boxes)
    if positives == 0:
        return 0.0
    detections = [record for record in records if record.predicted_box is not None]
    detections.sort(key=lambda r: (r.inlier_count, r.match_count), reverse=True)
    true_positives = 0
    false_positives = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for record in detections:
        if record.gt_boxes and record.best_iou >= threshold:
            true_positives += 1
        else:
            false_positives += 1
        precisions.append(true_positives / (true_positives + false_positives))
        recalls.append(true_positives / positives)
    total = 0.0
    for point in np.linspace(0.0, 1.0, 11):
        best = max(
            (
                precision
                for precision, recall in zip(precisions, recalls)
                if recall >= point
            ),
            default=0.0,
        )
        total += best
    return total / 11.0


def summarize(
    name: str,
    records: Sequence[SampleResult],
    iou_threshold: float,
) -> dict[str, object]:
    """Aggregate per-sample records into a model report.

    Args:
        name: Model name.
        records: Per-sample records.
        iou_threshold: IoU threshold that defines a hit.

    Returns:
        A JSON-ready report.
    """
    positives = [record for record in records if record.gt_boxes]
    hits = sum(1 for record in positives if record.hit)
    latencies = np.asarray([record.latency_ms for record in records], dtype=np.float64)
    mean_iou = (
        float(np.mean([record.best_iou for record in positives])) if positives else 0.0
    )
    false_positives = sum(
        1 for record in records if record.predicted_box is not None and not record.hit
    )
    reprojection = [
        record.reprojection_error
        for record in positives
        if record.reprojection_error is not None
    ]
    center_errors = [
        record.center_error for record in positives if record.center_error is not None
    ]
    return {
        "model": name,
        "samples": len(records),
        "positives": len(positives),
        "hit_rate": hits / len(positives) if positives else 0.0,
        "ap50": average_precision(records, iou_threshold),
        "mean_iou": mean_iou,
        "mean_matches": (
            float(np.mean([record.match_count for record in records]))
            if records
            else 0.0
        ),
        "mean_inliers": (
            float(np.mean([record.inlier_count for record in records]))
            if records
            else 0.0
        ),
        "mean_inlier_ratio": (
            float(np.mean([record.inlier_ratio for record in records]))
            if records
            else 0.0
        ),
        "mean_coverage": (
            float(np.mean([record.coverage for record in records])) if records else 0.0
        ),
        "median_reprojection_error": (
            float(np.median(reprojection)) if reprojection else None
        ),
        "valid_homography_rate": (
            sum(1 for record in positives if record.homography_valid) / len(positives)
            if positives
            else 0.0
        ),
        "mean_center_error": (float(np.mean(center_errors)) if center_errors else None),
        "stages": dict(Counter(record.stage for record in records)),
        "false_positives": false_positives,
        "latency_ms": {
            "mean": float(latencies.mean()) if len(latencies) else 0.0,
            "p50": float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
            "p95": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        },
        "per_sample": [record_dict(record) for record in records],
    }
