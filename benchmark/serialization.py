"""JSON serialization for per-sample records."""

from __future__ import annotations

from benchmark.types import SampleResult
from utils.dataset import Box


def record_dict(record: SampleResult) -> dict[str, object]:
    """Convert a per-sample record to JSON-ready data.

    Args:
        record: Per-sample record.

    Returns:
        A JSON-ready dictionary.
    """
    return {
        "dataset": record.dataset,
        "instance_id": record.instance_id,
        "query": record.query,
        "latency_ms": record.latency_ms,
        "match_count": record.match_count,
        "inlier_count": record.inlier_count,
        "best_iou": record.best_iou,
        "predicted_box": box_list(record.predicted_box),
        "gt_boxes": [box_list(box) for box in record.gt_boxes],
        "keypoints_ref": record.keypoints_ref,
        "keypoints_query": record.keypoints_query,
        "inlier_ratio": record.inlier_ratio,
        "coverage": record.coverage,
        "reprojection_error": record.reprojection_error,
        "homography_condition": record.homography_condition,
        "homography_valid": record.homography_valid,
        "box_area_ratio": record.box_area_ratio,
        "center_error": record.center_error,
        "stage": record.stage,
        "hit": record.hit,
    }


def box_list(box: Box | None) -> list[float] | None:
    """Convert an optional box to a JSON-ready list.

    Args:
        box: Optional box.

    Returns:
        A list of four floats, or ``None``.
    """
    return None if box is None else [float(value) for value in box]
