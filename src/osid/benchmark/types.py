"""Per-sample benchmark result."""

from __future__ import annotations

from dataclasses import dataclass

from osid.utils.dataset import Box


@dataclass
class SampleResult:
    """Per-sample benchmark record.

    Attributes:
        dataset: Dataset name.
        instance_id: Instance identifier.
        query: Query image path.
        latency_ms: Matching latency in milliseconds.
        match_count: Matches after the ratio test.
        inlier_count: RANSAC inliers.
        best_iou: Best IoU against the ground-truth boxes.
        predicted_box: Predicted query box, or ``None``.
        gt_boxes: Ground-truth query boxes.
        hit: Whether the best IoU reaches the hit threshold.
        keypoints_ref: Reference keypoint count.
        keypoints_query: Query keypoint count.
        inlier_ratio: Inliers divided by matches.
        coverage: Convex-hull area of inlier reference keypoints divided by
            the reference crop area.
        reprojection_error: Median inlier reprojection error in query pixels.
        homography_condition: Condition number of the estimated homography.
        homography_valid: Whether the homography and projected box pass sanity
            checks.
        box_area_ratio: Predicted box area divided by the query image area.
        center_error: Predicted-to-nearest-GT box center distance divided by
            the query diagonal.
        stage: Failure stage label.
    """

    dataset: str
    instance_id: str
    query: str
    latency_ms: float
    match_count: int
    inlier_count: int
    best_iou: float
    predicted_box: Box | None
    gt_boxes: tuple[Box, ...]
    hit: bool
    keypoints_ref: int = 0
    keypoints_query: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    reprojection_error: float | None = None
    homography_condition: float | None = None
    homography_valid: bool = False
    box_area_ratio: float | None = None
    center_error: float | None = None
    stage: str = "unknown"
