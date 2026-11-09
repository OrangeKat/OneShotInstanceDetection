"""Match overlay rendering for diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.models.feature_matching.base import MatchResult
from src.utils.diagnostics import Box

__all__ = ["render_overlay"]


def render_overlay(
    reference: np.ndarray,
    query: np.ndarray,
    result: MatchResult,
    box: Box | None,
    gt_boxes: Sequence[Box],
    diagnostics: dict[str, object],
    iou: float,
) -> np.ndarray:
    """Render reference/query with match lines and boxes.

    Args:
        reference: Reference crop.
        query: Query image.
        result: Matcher output.
        box: Predicted box.
        gt_boxes: Ground-truth boxes.
        diagnostics: Diagnostic fields.
        iou: Best IoU.

    Returns:
        BGR overlay image.
    """
    scale_ref = 800.0 / max(1, reference.shape[0])
    scale_query = 800.0 / max(1, query.shape[0])
    ref = cv2.resize(reference, None, fx=scale_ref, fy=scale_ref)
    qry = cv2.resize(query, None, fx=scale_query, fy=scale_query)
    gap = 20
    canvas = np.full(
        (800, ref.shape[1] + qry.shape[1] + gap, 3),
        255,
        dtype=np.uint8,
    )
    canvas[:, : ref.shape[1]] = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
    offset = ref.shape[1] + gap
    canvas[:, offset : offset + qry.shape[1]] = cv2.cvtColor(qry, cv2.COLOR_RGB2BGR)

    for index, (first, second) in enumerate(result.matches):
        source = result.keypoints0[first] * scale_ref
        target = result.keypoints1[second] * scale_query
        inlier = (
            bool(result.inlier_mask[index]) if result.inlier_mask is not None else False
        )
        color = (0, 180, 0) if inlier else (0, 0, 220)
        cv2.line(
            canvas,
            (int(source[0]), int(source[1])),
            (int(target[0] + offset), int(target[1])),
            color,
            1,
            cv2.LINE_AA,
        )
    for gt in gt_boxes:
        cv2.rectangle(
            canvas,
            (int(gt[0] * scale_query) + offset, int(gt[1] * scale_query)),
            (int(gt[2] * scale_query) + offset, int(gt[3] * scale_query)),
            (0, 200, 0),
            2,
        )
    if box is not None:
        cv2.rectangle(
            canvas,
            (int(box[0] * scale_query) + offset, int(box[1] * scale_query)),
            (int(box[2] * scale_query) + offset, int(box[3] * scale_query)),
            (0, 0, 255),
            2,
        )
    label = (
        f"{diagnostics['stage']} | matches={result.match_count} "
        f"inliers={result.inlier_count} | IoU={iou:.3f}"
    )
    cv2.putText(
        canvas,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas
