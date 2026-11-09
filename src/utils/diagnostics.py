"""Per-sample matching and geometry diagnostics."""

from __future__ import annotations

from typing import TypeAlias

import cv2
import numpy as np

from src.models.feature_matching.base import MatchResult

__all__ = ["Box", "failure_stage", "match_diagnostics"]

Box: TypeAlias = tuple[float, float, float, float]


def match_diagnostics(
    match: MatchResult,
    reference_shape: tuple[int, int],
    query_shape: tuple[int, int],
    box: Box | None,
    gt_boxes: tuple[Box, ...],
    best_iou: float,
    iou_threshold: float,
) -> dict[str, object]:
    """Compute per-sample matching and geometry diagnostics.

    Args:
        match: Matcher output.
        reference_shape: ``(height, width)`` of the reference crop.
        query_shape: ``(height, width)`` of the query image.
        box: Predicted query box, or ``None``.
        gt_boxes: Ground-truth query boxes.
        best_iou: Best IoU against the ground truth.
        iou_threshold: IoU threshold for a hit.

    Returns:
        Diagnostic fields.
    """
    reference_height, reference_width = reference_shape
    query_height, query_width = query_shape
    reference_area = max(1.0, reference_height * reference_width)
    query_area = max(1.0, query_height * query_width)
    inlier_ratio = match.inlier_count / match.match_count if match.match_count else 0.0

    coverage = 0.0
    reprojection_error: float | None = None
    condition: float | None = None
    valid = False
    if match.homography is not None and np.isfinite(match.homography).all():
        try:
            condition = float(np.linalg.cond(match.homography))
        except np.linalg.LinAlgError:
            condition = None
        if condition is not None and not np.isfinite(condition):
            condition = None
        if condition is not None:
            valid = condition < 1e8
        if match.matches:
            if match.inlier_mask is None:
                mask = np.ones(len(match.matches), dtype=bool)
            else:
                mask = np.asarray(match.inlier_mask, dtype=bool)
            source = match.keypoints0[[i for i, _ in match.matches]][mask]
            target = match.keypoints1[[j for _, j in match.matches]][mask]
            if len(source) >= 3:
                hull = cv2.convexHull(source.astype(np.float32))
                coverage = float(cv2.contourArea(hull)) / reference_area
            if len(source) >= 4:
                projected = cv2.perspectiveTransform(
                    source.reshape(1, -1, 2).astype(np.float64),
                    match.homography,
                ).reshape(-1, 2)
                distances = np.linalg.norm(projected - target, axis=1)
                if np.isfinite(distances).all():
                    reprojection_error = float(np.median(distances))

    box_area_ratio: float | None = None
    center_error: float | None = None
    if box is not None:
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        box_area_ratio = box_area / query_area
        if gt_boxes:
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            distances = [
                np.hypot(
                    center_x - (gt[0] + gt[2]) / 2.0,
                    center_y - (gt[1] + gt[3]) / 2.0,
                )
                for gt in gt_boxes
            ]
            diagonal = np.hypot(query_width, query_height)
            center_error = float(min(distances) / max(1.0, diagonal))
        if condition is not None:
            valid = valid and 1e-4 < box_area_ratio < 1e4
    elif condition is not None:
        valid = False

    return {
        "keypoints_ref": int(len(match.keypoints0)),
        "keypoints_query": int(len(match.keypoints1)),
        "inlier_ratio": float(inlier_ratio),
        "coverage": float(coverage),
        "reprojection_error": reprojection_error,
        "homography_condition": condition,
        "homography_valid": bool(valid),
        "box_area_ratio": box_area_ratio,
        "center_error": center_error,
        "stage": failure_stage(
            match,
            box,
            valid,
            gt_boxes,
            best_iou,
            iou_threshold,
        ),
    }


def failure_stage(
    match: MatchResult,
    box: Box | None,
    homography_valid: bool,
    gt_boxes: tuple[Box, ...],
    best_iou: float,
    iou_threshold: float,
) -> str:
    """Assign a failure stage to a sample.

    Args:
        match: Matcher output.
        box: Predicted box, or ``None``.
        homography_valid: Whether the homography passed sanity checks.
        gt_boxes: Ground-truth boxes.
        best_iou: Best IoU against the ground truth.
        iou_threshold: IoU threshold for a hit.

    Returns:
        Stage label.
    """
    if not gt_boxes:
        return "false_positive" if box is not None else "true_negative"
    if match.match_count == 0:
        return "no_matches"
    if match.match_count < 4:
        return "too_few_matches"
    if match.homography is None:
        return "ransac_failed"
    if not homography_valid:
        return "degenerate_homography"
    if box is None:
        return "no_prediction"
    if best_iou >= iou_threshold:
        return "hit"
    return "wrong_box"
