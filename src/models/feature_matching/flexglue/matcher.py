"""FlexGlue sparse-to-dense feature matching."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from src.models.feature_matching.base import (
    FeatureMatcher,
    MatchResult,
    estimate_homography,
)
from src.models.feature_matching.flexglue._vendor.flexglue import FlexGlue
from src.models.feature_matching.flexglue._vendor.flexglue.utils import (
    pack_matches,
    rbd,
)
from src.models.feature_matching.lightglue._vendor import ALIKED

__all__ = ["FlexGlueMatcher"]


class FlexGlueMatcher(FeatureMatcher):
    """FlexGlue matcher with an ALIKED keypoint detector."""

    name = "flexglue"

    def __init__(
        self,
        *,
        max_keypoints: int = 2048,
        ratio: float = 0.75,
        ransac_threshold: float = 3.0,
        device: str | None = None,
        max_side: int = 640,
    ) -> None:
        """Initialize FlexGlue and its ALIKED detector.

        Args:
            max_keypoints: Keypoint and match budget.
            ratio: Accepted for interface compatibility; unused.
            ransac_threshold: RANSAC reprojection threshold in pixels.
            device: ``cuda``, ``cpu``, or ``None`` for automatic selection.
            max_side: Longest image side fed to the detector and matcher;
                larger images are downscaled and keypoints are mapped back.
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
        self._detector = (
            ALIKED(origin="corner", max_num_keypoints=max_keypoints)
            .eval()
            .to(self.device)
        )
        self._model = FlexGlue().eval().to(self.device)

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference image.
            query: RGB query image.

        Returns:
            Matches and the estimated reference-to-query homography.
        """
        image0, scale0 = self._prepare(reference)
        image1, scale1 = self._prepare(query)
        with torch.inference_mode():
            prediction = self._model.match_pair(
                image0,
                image1,
                detector=self._detector,
                sparse=True,
            )
            packed = rbd(pack_matches(prediction))

        keypoints0 = packed["keypoints0"].detach().cpu().numpy()
        keypoints1 = packed["keypoints1"].detach().cpu().numpy()
        keypoints0 *= scale0
        keypoints1 *= scale1
        matches = packed["matches"].detach().cpu().numpy()
        scores = packed["scores"].detach().cpu().numpy()
        if len(matches) == 0:
            return MatchResult(keypoints0=keypoints0, keypoints1=keypoints1)

        if len(matches) > self.max_keypoints:
            keep = np.argsort(scores)[::-1][: self.max_keypoints]
            matches = matches[keep]
        pairs = tuple((int(first), int(second)) for first, second in matches)
        points0 = keypoints0[[first for first, _ in pairs]].astype(np.float32)
        points1 = keypoints1[[second for _, second in pairs]].astype(np.float32)
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

    def _prepare(self, image: np.ndarray) -> tuple[torch.Tensor, float]:
        """Convert an RGB uint8 image to a float tensor on the device.

        Args:
            image: RGB image.

        Returns:
            A ``(3, H, W)`` float tensor in ``[0, 1]`` on the target device,
            and the scale that maps network coordinates back to the original
            image size.
        """
        height, width = image.shape[:2]
        longest = max(height, width)
        scale = 1.0
        if longest > self.max_side:
            scale = self.max_side / longest
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device), 1.0 / scale
