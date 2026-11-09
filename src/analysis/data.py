"""Data loading and metrics for benchmark failure analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2

FAILURE_ORDER = [
    "hit",
    "wrong_box",
    "no_prediction",
    "false_positive",
    "true_negative",
]
FAILURE_LABELS = {
    "hit": "Hit",
    "wrong_box": "Wrong box",
    "no_prediction": "No prediction",
    "false_positive": "False positive",
    "true_negative": "True negative",
}
LOGO_SIZE_BINS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0]
LOGO_SIZE_LABELS = ["<0.5%", "0.5–1%", "1–2%", "2–5%", "5–10%", "10–20%", "≥20%"]
GT_COUNT_BINS = [1, 2, 3, 4, 10_000]
GT_COUNT_LABELS = ["1", "2", "3", "4+"]


@dataclass
class Row:
    """One model prediction for one reference/query pair.

    Attributes:
        model: Model name.
        class_name: Logo class / instance ID.
        query: Query file name.
        iou: Best IoU against the ground truth.
        hit: Whether the IoU reaches the hit threshold.
        matches: Matches after the ratio test.
        inliers: RANSAC inliers.
        latency_ms: Matching latency in milliseconds.
        gt_count: Number of ground-truth boxes.
        predicted: Whether a box was predicted.
        negative: Whether the query has no target boxes.
        logo_area_pct: Largest GT box area as a percentage of the query image.
        ref_area_pct: Reference crop area as a percentage of the reference image.
        inlier_ratio: Inliers divided by matches.
        stage: Failure stage label from the benchmark.
        coverage: Inlier convex-hull coverage of the reference crop.
        reprojection_error: Median inlier reprojection error in query pixels.
        homography_valid: Whether the homography passed sanity checks.
        keypoints_ref: Reference keypoint count.
        keypoints_query: Query keypoint count.
        box_area_ratio: Predicted box area divided by the query image area.
        center_error: Normalized predicted-to-nearest-GT center distance.
        query_area_px: Query image area in pixels.
        ref_area_px: Reference crop area in pixels.
        query_brightness: Mean grayscale value of the query image.
        query_contrast: Grayscale standard deviation of the query image.
        query_blur: Laplacian variance of the query image.
        ref_aspect: Reference crop width divided by height.
    """

    model: str
    class_name: str
    query: str
    iou: float
    hit: bool
    matches: int
    inliers: int
    latency_ms: float
    gt_count: int
    predicted: bool
    negative: bool
    logo_area_pct: float
    ref_area_pct: float
    inlier_ratio: float
    stage: str = ""
    coverage: float = 0.0
    reprojection_error: float | None = None
    homography_valid: bool = False
    keypoints_ref: int = 0
    keypoints_query: int = 0
    box_area_ratio: float | None = None
    center_error: float | None = None
    query_area_px: float = 0.0
    ref_area_px: float = 0.0
    query_brightness: float = 0.0
    query_contrast: float = 0.0
    query_blur: float = 0.0
    ref_aspect: float = 0.0

    @property
    def failure(self) -> str:
        """Return the failure category for this row."""
        if self.hit:
            return "hit"
        if self.negative:
            return "false_positive" if self.predicted else "true_negative"
        if not self.predicted:
            return "no_prediction"
        return "wrong_box"

    @property
    def query_area_mp(self) -> float:
        """Return the query image area in megapixels."""
        return self.query_area_px / 1e6


def load_rows(
    results_path: Path,
    manifest_path: Path,
) -> tuple[list[Row], Mapping[str, object]]:
    """Load and enrich per-sample results.

    Args:
        results_path: Results JSON.
        manifest_path: Benchmark manifest.

    Returns:
        Enriched rows and the result config.
    """
    payload = json.loads(results_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    by_key = {
        (str(record["instance_id"]), Path(str(record["query"])).name): record
        for record in manifest
    }
    size_cache: dict[Path, tuple[int, int, float, float, float]] = {}
    rows: list[Row] = []
    for report in payload["models"]:
        model = str(report["model"])
        for sample in report["per_sample"]:
            key = (str(sample["instance_id"]), Path(str(sample["query"])).name)
            record = by_key.get(key)
            if record is None:
                continue
            query_path = root / str(record["query"])
            width, height, brightness, contrast, blur = image_stats(
                query_path, size_cache
            )
            gt_boxes = sample["gt_boxes"]
            gt_count = len(gt_boxes)
            max_area = max(
                (
                    (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))
                    for box in gt_boxes
                ),
                default=0.0,
            )
            ref_path = root / str(record["reference"])
            ref_width, ref_height, _, _, _ = image_stats(ref_path, size_cache)
            ref_box = record.get("reference_box")
            ref_area = (
                (float(ref_box[2]) - float(ref_box[0]))
                * (float(ref_box[3]) - float(ref_box[1]))
                if ref_box is not None
                else float(ref_width * ref_height)
            )
            matches = int(sample["match_count"])
            inliers = int(sample["inlier_count"])
            rows.append(
                Row(
                    model=model,
                    class_name=str(sample["instance_id"]),
                    query=Path(str(sample["query"])).name,
                    iou=float(sample["best_iou"]),
                    hit=bool(sample["hit"]),
                    matches=matches,
                    inliers=inliers,
                    latency_ms=float(sample["latency_ms"]),
                    gt_count=gt_count,
                    predicted=sample["predicted_box"] is not None,
                    negative=gt_count == 0,
                    logo_area_pct=100.0 * max_area / max(1.0, width * height),
                    ref_area_pct=100.0 * ref_area / max(1.0, ref_width * ref_height),
                    inlier_ratio=float(
                        sample.get(
                            "inlier_ratio",
                            inliers / matches if matches else 0.0,
                        )
                    ),
                    stage=str(sample.get("stage", "")),
                    coverage=float(sample.get("coverage", 0.0)),
                    reprojection_error=_optional_float(
                        sample.get("reprojection_error")
                    ),
                    homography_valid=bool(sample.get("homography_valid", False)),
                    keypoints_ref=int(sample.get("keypoints_ref", 0)),
                    keypoints_query=int(sample.get("keypoints_query", 0)),
                    box_area_ratio=_optional_float(sample.get("box_area_ratio")),
                    center_error=_optional_float(sample.get("center_error")),
                    query_area_px=float(width * height),
                    ref_area_px=float(ref_area),
                    query_brightness=brightness,
                    query_contrast=contrast,
                    query_blur=blur,
                    ref_aspect=ref_width / max(1.0, ref_height),
                )
            )
    return rows, payload["config"]


def _optional_float(value: object) -> float | None:
    """Convert an optional JSON value to float.

    Args:
        value: JSON value or ``None``.

    Returns:
        Float value or ``None``.
    """
    return None if value is None else float(value)


def image_stats(
    path: Path,
    cache: dict[Path, tuple[int, int, float, float, float]],
) -> tuple[int, int, float, float, float]:
    """Return image size and simple appearance statistics, cached.

    Args:
        path: Image path.
        cache: Statistics cache.

    Returns:
        ``(width, height, brightness, contrast, laplacian_variance)``.

    Raises:
        FileNotFoundError: If the image cannot be decoded.
    """
    if path not in cache:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cache[path] = (
            width,
            height,
            float(gray.mean()),
            float(gray.std()),
            float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        )
    return cache[path]


def image_size(
    path: Path,
    cache: dict[Path, tuple[int, int, float, float, float]],
) -> tuple[int, int]:
    """Return image width and height, caching repeated reads.

    Args:
        path: Image path.
        cache: Statistics cache.

    Returns:
        ``(width, height)``.

    Raises:
        FileNotFoundError: If the image cannot be decoded.
    """
    width, height, _, _, _ = image_stats(path, cache)
    return width, height
