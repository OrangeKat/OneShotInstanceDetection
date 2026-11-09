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
        coarse_res: int = 560,
        upsample_res: int = 864,
    ) -> None:
        """Initialize RoMa.

        Args:
            max_keypoints: Number of correspondences drawn with RoMa's
                certainty-weighted, KDE-balanced sampling.
            ratio: Accepted for interface compatibility; unused.
            ransac_threshold: RANSAC reprojection threshold in pixels.
            device: ``cuda``, ``cpu``, or ``None`` for automatic selection.
            coarse_res: Coarse matching resolution. Lower values save VRAM.
            upsample_res: Dense upsampling resolution. Lower values save VRAM.
        """
        super().__init__(
            max_keypoints=max_keypoints,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.set_float32_matmul_precision("highest")
        self._model = roma_outdoor(
            str(self.device),
            coarse_res=coarse_res,
            upsample_res=upsample_res,
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
        matches, _ = self._model.sample(
            warp,
            certainty,
            num=self.max_keypoints,
        )
        matches = matches.detach().cpu().numpy()
        if len(matches) == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return MatchResult(keypoints0=empty, keypoints1=empty)

        reference_height, reference_width = reference.shape[:2]
        query_height, query_width = query.shape[:2]
        keypoints0 = self._to_pixels(
            matches[:, :2],
            reference_width,
            reference_height,
        )
        keypoints1 = self._to_pixels(
            matches[:, 2:4],
            query_width,
            query_height,
        )
        pairs = tuple((index, index) for index in range(len(matches)))
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
