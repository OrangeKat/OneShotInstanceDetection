"""LoFTR dense feature matching."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from kornia.color import rgb_to_grayscale
from kornia.feature import LoFTR

from src.models.feature_matching.base import (
    FeatureMatcher,
    MatchResult,
    estimate_homography,
)

__all__ = ["LoFTRMatcher"]


class LoFTRMatcher(FeatureMatcher):
    """LoFTR detector-free matcher with Kornia's pretrained weights."""

    name = "loftr"

    def __init__(
        self,
        *,
        max_keypoints: int = 2048,
        ratio: float = 0.75,
        ransac_threshold: float = 3.0,
        device: str | None = None,
        pretrained: str = "outdoor",
        max_side: int = 640,
    ) -> None:
        """Initialize LoFTR.

        Args:
            max_keypoints: Maximum matches kept for RANSAC.
            ratio: Accepted for interface compatibility; unused.
            ransac_threshold: RANSAC reprojection threshold in pixels.
            device: ``cuda``, ``cpu``, or ``None`` for automatic selection.
            pretrained: Kornia weight set, ``outdoor`` or ``indoor``.
            max_side: Longest image side fed to LoFTR; larger images are
                downscaled and keypoints are mapped back.
        """
        super().__init__(
            max_keypoints=max_keypoints,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_side = max_side
        self._model = LoFTR(pretrained=pretrained).eval().to(self.device)

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference image.
            query: RGB query image.

        Returns:
            Matches and the estimated reference-to-query homography.
        """
        image0, size0, scale0 = self._prepare(reference)
        image1, size1, scale1 = self._prepare(query)
        with torch.inference_mode():
            output = self._model({"image0": image0, "image1": image1})

        keypoints0 = output["keypoints0"].detach().cpu().numpy()
        keypoints1 = output["keypoints1"].detach().cpu().numpy()
        confidence = output["confidence"].detach().cpu().numpy()
        if len(keypoints0) == 0:
            return MatchResult(keypoints0=keypoints0, keypoints1=keypoints1)

        keypoints0 *= scale0
        keypoints1 *= scale1

        valid = (
            (keypoints0[:, 0] < size0[0])
            & (keypoints0[:, 1] < size0[1])
            & (keypoints1[:, 0] < size1[0])
            & (keypoints1[:, 1] < size1[1])
        )
        indices = np.flatnonzero(valid)
        if len(indices) > self.max_keypoints:
            keep = np.argsort(confidence[indices])[::-1][: self.max_keypoints]
            indices = indices[keep]
        if len(indices) == 0:
            return MatchResult(keypoints0=keypoints0, keypoints1=keypoints1)

        points0 = keypoints0[indices].astype(np.float32)
        points1 = keypoints1[indices].astype(np.float32)
        pairs = tuple((int(index), int(index)) for index in indices)
        homography, inlier_mask, inlier_count = estimate_homography(
            points0,
            points1,
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

    def _prepare(
        self,
        image: np.ndarray,
    ) -> tuple[torch.Tensor, tuple[int, int], float]:
        """Convert an RGB image to a padded grayscale tensor.

        Args:
            image: RGB image.

        Returns:
            A ``(1, 1, H, W)`` tensor, the original ``(width, height)``, and the
            scale that maps network coordinates back to the original pixels.
        """
        original_height, original_width = image.shape[:2]
        height, width = original_height, original_width
        longest = max(height, width)
        scale = 1.0
        if longest > self.max_side:
            scale = self.max_side / longest
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = image.shape[:2]

        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = rgb_to_grayscale(tensor.unsqueeze(0))
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        if pad_height or pad_width:
            tensor = torch.nn.functional.pad(
                tensor,
                (0, pad_width, 0, pad_height),
                mode="replicate",
            )
        return (
            tensor.to(self.device),
            (original_width, original_height),
            1.0 / scale,
        )
