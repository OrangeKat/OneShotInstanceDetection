"""Match overlay rendering for diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.models.feature_matching.base import MatchResult
from src.utils.diagnostics import Box

__all__ = ["render_overlay"]

MAX_INLIER_LINES = 60
MAX_OUTLIER_LINES = 15
MAX_SIDE = (760, 600)


def render_overlay(
    reference: np.ndarray,
    query: np.ndarray,
    result: MatchResult,
    box: Box | None,
    gt_boxes: Sequence[Box],
    diagnostics: dict[str, object],
    iou: float,
) -> np.ndarray:
    """Render reference/query with sampled match lines and boxes.

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
    ref, scale_ref = _fit(reference)
    qry, scale_query = _fit(query)
    gap = 20
    header = 34
    height = max(ref.shape[0], qry.shape[0])
    canvas = np.full(
        (height + header, ref.shape[1] + qry.shape[1] + gap, 3),
        255,
        dtype=np.uint8,
    )
    canvas[header : header + ref.shape[0], : ref.shape[1]] = cv2.cvtColor(
        ref, cv2.COLOR_RGB2BGR
    )
    offset = ref.shape[1] + gap
    canvas[
        header : header + qry.shape[0],
        offset : offset + qry.shape[1],
    ] = cv2.cvtColor(qry, cv2.COLOR_RGB2BGR)

    mask = (
        np.asarray(result.inlier_mask, dtype=bool)
        if result.inlier_mask is not None
        else np.ones(len(result.matches), dtype=bool)
    )
    inliers = [index for index, keep in enumerate(mask) if keep]
    outliers = [index for index, keep in enumerate(mask) if not keep]
    for index in _sample(inliers, MAX_INLIER_LINES):
        _draw_match(
            canvas,
            result,
            index,
            scale_ref,
            scale_query,
            offset,
            header,
            (0, 180, 0),
        )
    for index in _sample(outliers, MAX_OUTLIER_LINES):
        _draw_match(
            canvas,
            result,
            index,
            scale_ref,
            scale_query,
            offset,
            header,
            (0, 0, 220),
        )

    for gt in gt_boxes:
        _draw_box(canvas, gt, scale_query, offset, header, (0, 200, 0))
    if box is not None:
        _draw_box(canvas, box, scale_query, offset, header, (0, 0, 255))

    label = (
        f"{diagnostics['stage']} | matches={result.match_count} "
        f"inliers={result.inlier_count} | IoU={iou:.3f}"
    )
    cv2.putText(
        canvas,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green = inlier, red = outlier | green box = GT, red box = prediction",
        (max(10, canvas.shape[1] // 2 - 260), 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _fit(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize an image to fit the overlay panel.

    Args:
        image: RGB image.

    Returns:
        Resized image and the scale factor.
    """
    height, width = image.shape[:2]
    scale = min(MAX_SIDE[0] / max(1, width), MAX_SIDE[1] / max(1, height))
    if abs(scale - 1.0) < 1e-6:
        return image, 1.0
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    return resized, scale


def _sample(indices: Sequence[int], limit: int) -> list[int]:
    """Evenly sample at most ``limit`` indices.

    Args:
        indices: Match indices.
        limit: Maximum number of indices.

    Returns:
        Sampled indices.
    """
    if len(indices) <= limit:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return [indices[position] for position in positions]


def _draw_match(
    canvas: np.ndarray,
    result: MatchResult,
    index: int,
    scale_ref: float,
    scale_query: float,
    offset: int,
    header: int,
    color: tuple[int, int, int],
) -> None:
    """Draw one match line and its endpoints.

    Args:
        canvas: Output canvas.
        result: Matcher output.
        index: Match index.
        scale_ref: Reference scale.
        scale_query: Query scale.
        offset: Query x offset.
        header: Canvas header height.
        color: BGR color.
    """
    first, second = result.matches[index]
    source = result.keypoints0[first] * scale_ref
    target = result.keypoints1[second] * scale_query
    start = (int(source[0]), int(source[1] + header))
    end = (int(target[0] + offset), int(target[1] + header))
    cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, start, 2, color, -1)
    cv2.circle(canvas, end, 2, color, -1)


def _draw_box(
    canvas: np.ndarray,
    box: Box,
    scale: float,
    offset: int,
    header: int,
    color: tuple[int, int, int],
) -> None:
    """Draw one box on the query panel.

    Args:
        canvas: Output canvas.
        box: ``[x1, y1, x2, y2]`` box.
        scale: Query scale.
        offset: Query x offset.
        header: Canvas header height.
        color: BGR color.
    """
    cv2.rectangle(
        canvas,
        (int(box[0] * scale) + offset, int(box[1] * scale) + header),
        (int(box[2] * scale) + offset, int(box[3] * scale) + header),
        color,
        2,
    )
