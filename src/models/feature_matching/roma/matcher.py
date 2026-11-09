"""RoMa dense feature matching."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.models.feature_matching.base import (
    FeatureMatcher,
    MatchResult,
    estimate_homography,
)
from src.models.feature_matching.roma._vendor.romatch.models.model_zoo import (
    roma_outdoor,
)

__all__ = ["RoMaMatcher"]


class RoMaMatcher(FeatureMatcher):
    """RoMa v1 dense matcher with the official outdoor weights."""

    name = "roma"

    def __init__(
        self,
        *,
        max_keypoints: int = 2048,
        ratio: float = 0.75,
        ransac_threshold: float = 3.0,
        device: str | None = None,
        certainty_threshold: float = 0.05,
    ) -> None:
        """Initialize RoMa.

        Args:
            max_keypoints: Maximum sampled correspondences kept for RANSAC.
            ratio: Accepted for interface compatibility; unused.
            ransac_threshold: RANSAC reprojection threshold in pixels.
            device: ``cuda``, ``cpu``, or ``None`` for automatic selection.
            certainty_threshold: Minimum RoMa certainty for a sampled match.
        """
        super().__init__(
            max_keypoints=max_keypoints,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.certainty_threshold = certainty_threshold
        torch.set_float32_matmul_precision("highest")
        self._model = roma_outdoor(
            str(self.device),
            use_custom_corr=False,
        )

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference image.
            query: RGB query image.

        Returns:
            Matches and the estimated reference-to-query homography.
        """
        image0 = Image.fromarray(reference)
        image1 = Image.fromarray(query)
        warp, certainty = self._model.match(image0, image1)
        warp = warp.detach().cpu().numpy()
        certainty = certainty.detach().cpu().numpy()
        if warp.ndim == 4:
            warp = warp[0]
        if certainty.ndim == 3:
            certainty = certainty[0]

        height, width = warp.shape[:2]
        if certainty.shape[-1] == 2 * width:
            certainty = certainty[..., :width]
        points0 = warp[..., :2].reshape(-1, 2)
        points1 = warp[..., 2:4].reshape(-1, 2)
        scores = certainty.reshape(-1)

        valid = (
            (scores >= self.certainty_threshold)
            & (np.abs(points1[:, 0]) <= 1.0)
            & (np.abs(points1[:, 1]) <= 1.0)
        )
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(keypoints0=empty, keypoints1=empty)

        if len(indices) > self.max_keypoints:
            keep = np.argsort(scores[indices])[::-1][: self.max_keypoints]
            indices = indices[keep]

        reference_height, reference_width = reference.shape[:2]
        query_height, query_width = query.shape[:2]
        keypoints0 = self._to_pixels(
            points0[indices],
            reference_width,
            reference_height,
        )
        keypoints1 = self._to_pixels(
            points1[indices],
            query_width,
            query_height,
        )
        pairs = tuple((index, index) for index in range(len(indices)))
        homography, inlier_mask, inlier_count = estimate_homography(
            keypoints0,
            keypoints1,
            self.ransac_threshold,
        )
        return MatchResult(
            keypoints0=keypoints0,
            keypoints1=keypoints1,
            matches=pairs,
            homography=homography,
            inlier_mask=inlier_mask,
            match_count=len(pairs),
            inlier_count=inlier_count,
        )

    @staticmethod
    def _to_pixels(
        points: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Convert RoMa normalized coordinates to pixels.

        Args:
            points: Normalized ``(x, y)`` points in ``[-1, 1]``.
            width: Image width.
            height: Image height.

        Returns:
            Pixel coordinates.
        """
        pixels = np.empty_like(points, dtype=np.float32)
        pixels[:, 0] = (points[:, 0] + 1.0) * width / 2.0 - 0.5
        pixels[:, 1] = (points[:, 1] + 1.0) * height / 2.0 - 0.5
        return pixels
