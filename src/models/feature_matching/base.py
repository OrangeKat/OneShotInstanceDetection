"""Shared types and helpers for feature-matching models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "FeatureMatcher",
    "MatchResult",
    "estimate_homography",
    "keypoint_points",
    "limit_keypoints",
    "ratio_test",
    "to_gray",
]


@dataclass
class MatchResult:
    """Output of a two-image feature matcher.

    Attributes:
        keypoints0: Reference keypoints, shape ``(N, 2)``.
        keypoints1: Query keypoints, shape ``(M, 2)``.
        matches: Matched index pairs after the ratio test.
        homography: Reference-to-query homography, or ``None``.
        inlier_mask: RANSAC inlier mask aligned with ``matches``.
        match_count: Number of matches after the ratio test.
        inlier_count: Number of RANSAC inliers.
    """

    keypoints0: np.ndarray
    keypoints1: np.ndarray
    matches: tuple[tuple[int, int], ...] = ()
    homography: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    match_count: int = 0
    inlier_count: int = 0


class FeatureMatcher(ABC):
    """Base class for a reference-to-query feature matcher.

    Attributes:
        name: Short model name used in benchmark output.
        max_keypoints: Keypoint budget per image.
        ratio: Lowe ratio-test threshold.
        ransac_threshold: RANSAC reprojection threshold in pixels.
    """

    name: str

    def __init__(
        self,
        *,
        max_keypoints: int = 2048,
        ratio: float = 0.75,
        ransac_threshold: float = 3.0,
    ) -> None:
        """Initialize shared matcher settings.

        Args:
            max_keypoints: Keypoint budget per image.
            ratio: Lowe ratio-test threshold.
            ransac_threshold: RANSAC reprojection threshold in pixels.
        """
        self.max_keypoints = max_keypoints
        self.ratio = ratio
        self.ransac_threshold = ransac_threshold

    @abstractmethod
    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference image.
            query: RGB query image.

        Returns:
            Matches and the estimated reference-to-query homography.
        """

    def _result(
        self,
        keypoints0: Sequence[cv2.KeyPoint],
        keypoints1: Sequence[cv2.KeyPoint],
        matches: tuple[tuple[int, int], ...],
    ) -> MatchResult:
        """Estimate a homography and package a match result.

        Args:
            keypoints0: Reference keypoints.
            keypoints1: Query keypoints.
            matches: Matched index pairs after the ratio test.

        Returns:
            A populated match result.
        """
        points0 = keypoint_points(keypoints0)
        points1 = keypoint_points(keypoints1)
        if not matches:
            return MatchResult(keypoints0=points0, keypoints1=points1)

        source = points0[[index for index, _ in matches]]
        target = points1[[index for _, index in matches]]
        homography, inlier_mask, inlier_count = estimate_homography(
            source,
            target,
            self.ransac_threshold,
        )
        return MatchResult(
            keypoints0=points0,
            keypoints1=points1,
            matches=matches,
            homography=homography,
            inlier_mask=inlier_mask,
            match_count=len(matches),
            inlier_count=inlier_count,
        )


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to 8-bit grayscale.

    Args:
        image: RGB or already grayscale image.

    Returns:
        An 8-bit grayscale image.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def keypoint_points(keypoints: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """Convert OpenCV keypoints to an ``(N, 2)`` float32 array.

    Args:
        keypoints: OpenCV keypoints.

    Returns:
        An array of ``(x, y)`` coordinates.
    """
    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)


def limit_keypoints(
    keypoints: Sequence[cv2.KeyPoint],
    descriptors: np.ndarray | None,
    limit: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Keep the strongest keypoints up to ``limit``.

    Args:
        keypoints: OpenCV keypoints.
        descriptors: Descriptor rows aligned with ``keypoints``.
        limit: Maximum number of keypoints to keep.

    Returns:
        The filtered keypoints and descriptors.
    """
    if descriptors is None or len(keypoints) <= limit:
        return list(keypoints), descriptors
    order = np.argsort([keypoint.response for keypoint in keypoints])[::-1][:limit]
    return [keypoints[index] for index in order], descriptors[order]


def ratio_test(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    ratio: float,
) -> tuple[tuple[int, int], ...]:
    """Apply Lowe's ratio test to k-nearest-neighbour matches.

    Args:
        knn_matches: Two-nearest-neighbour matches per reference descriptor.
        ratio: Maximum distance ratio between the best and second best match.

    Returns:
        Matched ``(reference_index, query_index)`` pairs.
    """
    pairs: list[tuple[int, int]] = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        best, second = candidates[0], candidates[1]
        if best.distance < ratio * second.distance:
            pairs.append((best.queryIdx, best.trainIdx))
    return tuple(pairs)


def estimate_homography(
    source: np.ndarray,
    target: np.ndarray,
    ransac_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Estimate a homography with RANSAC.

    Args:
        source: Source points, shape ``(N, 2)``.
        target: Target points, shape ``(N, 2)``.
        ransac_threshold: Reprojection threshold in pixels.

    Returns:
        The homography, the inlier mask, and the inlier count.
    """
    if len(source) < 4:
        return None, None, 0
    homography, mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        ransac_threshold,
        maxIters=5000,
        confidence=0.999,
    )
    if homography is None or mask is None:
        return None, None, 0
    inliers = mask.ravel().astype(bool)
    return homography, inliers, int(inliers.sum())
