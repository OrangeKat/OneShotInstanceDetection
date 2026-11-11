"""HPatches sequence loading and native matching metrics."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.models.feature_matching.base import FeatureMatcher

TARGETS = range(2, 7)
THRESHOLDS = (1, 3, 5, 10)


@dataclass
class HPatchesPair:
    """One reference/target pair from an HPatches sequence.

    Attributes:
        sequence: Sequence name.
        target: Target image number (2-6).
        split: ``"illumination"``, ``"viewpoint"``, or ``"unknown"``.
        reference_path: Path to ``1.ppm``.
        query_path: Path to the target image.
        homography: Reference-to-target homography.
    """

    sequence: str
    target: int
    split: str
    reference_path: Path
    query_path: Path
    homography: np.ndarray


def load_pairs(sequences_dir: Path, limit: int | None = None) -> list[HPatchesPair]:
    """Load HPatches reference/target pairs.

    Args:
        sequences_dir: Extracted ``hpatches-sequences-release`` directory.
        limit: Optional limit on the number of sequences.

    Returns:
        Pairs in sequence and target order.
    """
    sequence_dirs = sorted(path for path in sequences_dir.iterdir() if path.is_dir())
    if limit is not None:
        sequence_dirs = sequence_dirs[:limit]
    pairs: list[HPatchesPair] = []
    for sequence_dir in sequence_dirs:
        reference = sequence_dir / "1.ppm"
        if not reference.exists():
            continue
        split = _split_name(sequence_dir.name)
        for target in TARGETS:
            query = sequence_dir / f"{target}.ppm"
            homography_path = sequence_dir / f"H_1_{target}"
            if not query.exists() or not homography_path.exists():
                continue
            homography = np.loadtxt(homography_path)
            if homography.shape != (3, 3):
                continue
            pairs.append(
                HPatchesPair(
                    sequence=sequence_dir.name,
                    target=target,
                    split=split,
                    reference_path=reference,
                    query_path=query,
                    homography=homography,
                )
            )
    return pairs


def evaluate_model(
    matcher: FeatureMatcher,
    pairs: list[HPatchesPair],
    thresholds: tuple[int, ...] = THRESHOLDS,
    progress: Callable[[int], None] | None = None,
) -> dict[str, object]:
    """Evaluate a matcher with MMA, HEA, and corner error.

    Args:
        matcher: Feature matcher to evaluate.
        pairs: HPatches reference/target pairs.
        thresholds: Pixel thresholds for MMA and HEA.
        progress: Optional callback called with the completed pair count.

    Returns:
        Aggregate metrics and per-pair records.
    """
    mma_values: dict[int, list[float]] = {threshold: [] for threshold in thresholds}
    hea_values: dict[int, list[bool]] = {threshold: [] for threshold in thresholds}
    corner_errors: list[float] = []
    per_pair: list[dict[str, object]] = []
    latencies: list[float] = []
    total_matches = 0
    total_inliers = 0
    pairs_with_matches = 0
    pairs_with_homography = 0

    for index, pair in enumerate(pairs, start=1):
        reference = _load_rgb(pair.reference_path)
        query = _load_rgb(pair.query_path)
        height, width = reference.shape[:2]
        start = time.perf_counter()
        result = matcher.match(reference, query)
        latency = (time.perf_counter() - start) * 1000.0
        latencies.append(latency)
        total_matches += result.match_count
        total_inliers += result.inlier_count

        pair_mma: dict[str, float | None] = {
            str(threshold): None for threshold in thresholds
        }
        if result.match_count:
            pairs_with_matches += 1
            reference_points = result.keypoints0[[i for i, _ in result.matches]]
            query_points = result.keypoints1[[j for _, j in result.matches]]
            projected = cv2.perspectiveTransform(
                reference_points.reshape(1, -1, 2), pair.homography
            ).reshape(-1, 2)
            distances = np.linalg.norm(projected - query_points, axis=1)
            for threshold in thresholds:
                accuracy = float((distances <= threshold).mean())
                pair_mma[str(threshold)] = accuracy
                mma_values[threshold].append(accuracy)

        corner: float | None = None
        if result.homography is not None:
            pairs_with_homography += 1
            corner = corner_error(
                pair.homography,
                result.homography,
                width,
                height,
            )
            corner_errors.append(corner)
        for threshold in thresholds:
            hea_values[threshold].append(corner is not None and corner <= threshold)

        per_pair.append(
            {
                "sequence": pair.sequence,
                "target": pair.target,
                "split": pair.split,
                "matches": result.match_count,
                "inliers": result.inlier_count,
                "mma": pair_mma,
                "corner_error": corner,
                "latency_ms": latency,
            }
        )
        if progress is not None:
            progress(index)

    return {
        "model": matcher.name,
        "pairs": len(pairs),
        "pairs_with_matches": pairs_with_matches,
        "pairs_with_homography": pairs_with_homography,
        "mma": {
            str(threshold): _mean(mma_values[threshold]) for threshold in thresholds
        },
        "hea": {
            str(threshold): _mean(hea_values[threshold]) for threshold in thresholds
        },
        "auc": homography_auc(
            [pair["corner_error"] for pair in per_pair],
            thresholds,
        ),
        "corner_error": {
            "mean": float(np.mean(corner_errors)) if corner_errors else None,
            "median": float(np.median(corner_errors)) if corner_errors else None,
            "p90": float(np.percentile(corner_errors, 90)) if corner_errors else None,
        },
        "mean_matches": total_matches / len(pairs) if pairs else 0.0,
        "mean_inliers": total_inliers / len(pairs) if pairs else 0.0,
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "per_pair": per_pair,
    }


def homography_auc(
    corner_errors: Sequence[float | None],
    thresholds: Sequence[int] = THRESHOLDS,
) -> dict[str, float | None]:
    """Compute homography AUC from per-pair corner errors.

    Args:
        corner_errors: Mean four-corner errors; ``None`` marks a missing
            homography.
        thresholds: Pixel thresholds for the cumulative error curve.

    Returns:
        AUC values keyed by threshold.
    """
    if not corner_errors:
        return {str(threshold): None for threshold in thresholds}
    errors = np.asarray(
        [error if error is not None else np.inf for error in corner_errors],
        dtype=np.float64,
    )
    result: dict[str, float | None] = {}
    for threshold in thresholds:
        if threshold <= 0:
            result[str(threshold)] = None
            continue
        grid = np.linspace(0.0, float(threshold), 1001)
        recall = (errors[:, None] <= grid[None, :]).mean(axis=0)
        result[str(threshold)] = float(np.trapezoid(recall, grid) / threshold)
    return result


def corner_error(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    width: int,
    height: int,
) -> float:
    """Compute mean corner error between two homographies.

    Args:
        ground_truth: Reference-to-query ground-truth homography.
        predicted: Estimated reference-to-query homography.
        width: Reference image width.
        height: Reference image height.

    Returns:
        Mean Euclidean distance between the four projected corners.
    """
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    gt_corners = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), ground_truth
    ).reshape(-1, 2)
    pred_corners = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), predicted
    ).reshape(-1, 2)
    return float(np.linalg.norm(gt_corners - pred_corners, axis=1).mean())


def _split_name(sequence: str) -> str:
    """Return the HPatches split for a sequence name.

    Args:
        sequence: Sequence folder name.

    Returns:
        Split name.
    """
    if sequence.startswith("i_"):
        return "illumination"
    if sequence.startswith("v_"):
        return "viewpoint"
    return "unknown"


def _mean(values: list[float] | list[bool]) -> float | None:
    """Return the mean of a list, or ``None`` when empty.

    Args:
        values: Numeric or boolean values.

    Returns:
        Mean value or ``None``.
    """
    if not values:
        return None
    return float(np.mean(values))


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB.

    Args:
        path: Image path.

    Returns:
        RGB image.

    Raises:
        FileNotFoundError: If the image cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
