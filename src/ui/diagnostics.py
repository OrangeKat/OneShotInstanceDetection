"""Per-pair diagnostic dump backing the inspector.

One dump answers four questions about a single reference/query pair: what the
matcher returned before RANSAC, which reference keypoint each query point came
from, how far each match lands from where the estimated homography says it
should, and where each side concentrated its attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.models.feature_matching.base import FeatureMatcher, MatchResult
from src.utils.dataset import InstanceDataset, InstanceSample

__all__ = ["build_dump", "density_grid", "oci_heatmap"]

DENSITY_BINS = 32
OCLUSION_FILL = 127


def density_grid(points: np.ndarray, width: int, height: int) -> list[list[float]]:
    """Bin points into a normalized occupancy grid.

    Args:
        points: ``(N, 2)`` pixel coordinates.
        width: Image width.
        height: Image height.

    Returns:
        ``DENSITY_BINS x DENSITY_BINS`` grid of counts normalized to ``[0, 1]``.
    """
    grid, _, _ = np.histogram2d(
        np.clip(points[:, 1], 0, max(height - 1, 0)) if len(points) else [],
        np.clip(points[:, 0], 0, max(width - 1, 0)) if len(points) else [],
        bins=DENSITY_BINS,
        range=[[0, max(height, 1)], [0, max(width, 1)]],
    )
    peak = float(grid.max()) if grid.size else 0.0
    if peak <= 0:
        return [[0.0] * DENSITY_BINS for _ in range(DENSITY_BINS)]
    return (grid / peak).round(4).tolist()


def _residuals(
    matches: Sequence[tuple[int, int]],
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    homography: np.ndarray | None,
) -> list[float | None]:
    """Return the distance between each match and its projected reference point.

    Args:
        matches: Matched index pairs.
        keypoints0: Reference keypoints.
        keypoints1: Query keypoints.
        homography: Estimated reference-to-query homography, or ``None``.

    Returns:
        One residual per match in query pixels, or ``None`` without a homography.
    """
    if homography is None or not matches:
        return [None] * len(matches)
    source = keypoints0[[index0 for index0, _ in matches]]
    target = keypoints1[[index1 for _, index1 in matches]]
    projected = _project(source, homography)
    residuals = np.linalg.norm(projected - target, axis=1)
    return [round(float(value), 2) for value in residuals]


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project points through a homography.

    Args:
        points: ``(N, 2)`` coordinates.
        homography: ``3 x 3`` homography.

    Returns:
        ``(N, 2)`` projected coordinates.
    """
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1))],
        axis=1,
    )
    projected = homogeneous @ homography.T
    return projected[:, :2] / np.clip(projected[:, 2:3], 1e-9, None)


def build_dump(
    dataset: InstanceDataset,
    sample: InstanceSample,
    result: MatchResult,
    *,
    model: str,
    index: int,
) -> dict[str, object]:
    """Assemble the JSON payload for one pair and one model.

    Args:
        dataset: Manifest-backed dataset, used to crop the reference.
        sample: Pair to inspect, with images already loaded.
        result: Matcher output for that pair.
        model: Model name.
        index: Manifest index of the pair.

    Returns:
        JSON-serializable dump describing images, correspondences, and maps.
    """
    reference = dataset.reference_crop(sample)
    query = sample.query_image
    if query is None:
        raise ValueError("query image is not loaded")
    reference_height, reference_width = reference.shape[:2]
    query_height, query_width = query.shape[:2]

    matches = list(result.matches)
    scores = list(result.scores)
    mask = (
        [bool(value) for value in result.inlier_mask]
        if result.inlier_mask is not None
        else [False] * len(matches)
    )
    residuals = _residuals(
        matches, result.keypoints0, result.keypoints1, result.homography
    )

    points0 = np.asarray(result.keypoints0, dtype=float).reshape(-1, 2)
    points1 = np.asarray(result.keypoints1, dtype=float).reshape(-1, 2)
    correspondences = []
    for position, (index0, index1) in enumerate(matches):
        correspondences.append(
            {
                "ref": int(index0),
                "query": int(index1),
                "x0": round(float(points0[index0, 0]), 2),
                "y0": round(float(points0[index0, 1]), 2),
                "x1": round(float(points1[index1, 0]), 2),
                "y1": round(float(points1[index1, 1]), 2),
                "score": round(float(scores[position]), 4)
                if position < len(scores)
                else None,
                "inlier": mask[position] if position < len(mask) else False,
                "residual": residuals[position] if position < len(residuals) else None,
            }
        )

    matched_reference = (
        points0[[index0 for index0, _ in matches]] if matches else points0[:0]
    )
    matched_query = (
        points1[[index1 for _, index1 in matches]] if matches else points1[:0]
    )

    return {
        "model": model,
        "index": index,
        "instance_id": sample.instance_id,
        "query": sample.query_path.name,
        "reference_url": "/img/reference",
        "query_url": "/img/query",
        "reference_size": [reference_width, reference_height],
        "query_size": [query_width, query_height],
        "gt_boxes": [list(box) for box in sample.query_boxes],
        "matches": correspondences,
        "match_count": len(matches),
        "inlier_count": int(sum(mask)),
        "has_scores": bool(scores),
        "homography": None if result.homography is None else result.homography.tolist(),
        "reference_density": density_grid(
            np.asarray(matched_reference, dtype=float),
            reference_width,
            reference_height,
        ),
        "query_density": density_grid(
            np.asarray(matched_query, dtype=float),
            query_width,
            query_height,
        ),
    }


def oci_heatmap(
    dataset: InstanceDataset,
    sample: InstanceSample,
    matcher: FeatureMatcher,
    baseline: MatchResult,
    *,
    grid: int = 6,
) -> dict[str, object]:
    """Measure which reference regions the matches depend on.

    Each cell of a ``grid x grid`` overlay is filled with flat grey and the
    matcher re-run. The value reported per cell is the share of baseline matches
    that survive, so high values mark regions the correspondence set relies on.

    Args:
        dataset: Manifest-backed dataset.
        sample: Pair to inspect, with images already loaded.
        matcher: Matcher instance to re-run.
        baseline: Unmodified match result for the same pair.
        grid: Cells per side.

    Returns:
        JSON-serializable heatmap with survival shares and the cell geometry.
    """
    reference = dataset.reference_crop(sample)
    query = sample.query_image
    if query is None:
        raise ValueError("query image is not loaded")
    height, width = reference.shape[:2]
    base_pairs = _match_keys(baseline)
    if not base_pairs:
        return {"grid": grid, "cells": [], "note": "model produced no matches"}

    values: list[list[float]] = []
    for row in range(grid):
        row_values: list[float] = []
        for column in range(grid):
            occluded = reference.copy()
            y0 = int(height * row / grid)
            y1 = int(height * (row + 1) / grid)
            x0 = int(width * column / grid)
            x1 = int(width * (column + 1) / grid)
            occluded[y0:y1, x0:x1] = OCLUSION_FILL
            result = matcher.match(occluded, query)
            kept = len(base_pairs & _match_keys(result))
            row_values.append(round(kept / len(base_pairs), 3))
        values.append(row_values)
    return {"grid": grid, "size": [width, height], "cells": values}


def _match_keys(result: MatchResult) -> set[tuple[int, int]]:
    """Return the matched index pairs of a result.

    Args:
        result: Matcher output.

    Returns:
        Set of ``(reference_index, query_index)`` pairs.
    """
    return {(int(index0), int(index1)) for index0, index1 in result.matches}


def cache_path(cache_dir: Path, model: str, index: int) -> Path:
    """Return the on-disk path for a cached dump.

    Args:
        cache_dir: Cache root.
        model: Model name.
        index: Manifest index.

    Returns:
        Path to the JSON dump.
    """
    return cache_dir / f"{index:05d}-{model}.json"
