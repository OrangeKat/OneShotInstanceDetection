"""ORB feature matching."""

from __future__ import annotations

import cv2
import numpy as np

from src.models.feature_matching.base import (
    FeatureMatcher,
    MatchResult,
    ratio_test,
    to_gray,
)

__all__ = ["ORBMatcher"]


class ORBMatcher(FeatureMatcher):
    """ORB detector/descriptor with BF-Hamming matching and RANSAC."""

    name = "orb"

    def __init__(
        self,
        *,
        max_keypoints: int = 2048,
        ratio: float = 0.75,
        ransac_threshold: float = 3.0,
    ) -> None:
        """Initialize the ORB matcher.

        Args:
            max_keypoints: Keypoint budget per image.
            ratio: Lowe ratio-test threshold.
            ransac_threshold: RANSAC reprojection threshold in pixels.
        """
        super().__init__(
            max_keypoints=max_keypoints,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
        )
        self._detector = cv2.ORB_create(nfeatures=max_keypoints)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference image.
            query: RGB query image.

        Returns:
            Matches and the estimated reference-to-query homography.
        """
        keypoints0, descriptors0 = self._detector.detectAndCompute(
            to_gray(reference), None
        )
        keypoints1, descriptors1 = self._detector.detectAndCompute(to_gray(query), None)
        if descriptors0 is None or descriptors1 is None:
            return self._result(keypoints0, keypoints1, ())

        matches = ratio_test(
            self._matcher.knnMatch(descriptors0, descriptors1, k=2),
            self.ratio,
        )
        return self._result(keypoints0, keypoints1, matches)
