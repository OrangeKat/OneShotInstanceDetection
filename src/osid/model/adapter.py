"""Benchmark adapter: the detector plus a plain geometric matcher.

The model itself is per-image. The benchmark asks for a function of two images,
so this module supplies the simplest matcher that turns two detection sets into
a ``MatchResult``: mutual nearest neighbours, a ratio test, and RANSAC.

It is deliberately not part of the model. Replace it with a learned matcher, an
optimal-transport matcher, or anything else and the detector is unchanged.
"""

from __future__ import annotations

import numpy as np

from osid.benchmark.matchers.base import MatchResult, estimate_homography
from osid.model.config import ModelConfig
from osid.model.detector import Detections, DetectorDescriptor
from osid.model.matcher import match_descriptors

__all__ = ["BenchmarkAdapter"]


class BenchmarkAdapter:
    """Runs the detector on both images and matches the descriptors.

    Attributes:
        name: Benchmark identifier.
        detector: The per-image model.
        ransac_threshold: Reprojection threshold for the homography.
    """

    name = "osid_boundary"

    def __init__(
        self,
        detector: DetectorDescriptor | None = None,
        *,
        device: str | None = None,
        max_keypoints: int | None = None,
        ransac_threshold: float = 3.0,
        config: ModelConfig | None = None,
    ) -> None:
        """Build the adapter.

        Args:
            detector: Existing detector to reuse; a new one is built otherwise.
            device: Device for a newly built detector.
            max_keypoints: Keypoint budget for a newly built detector.
            ransac_threshold: Reprojection threshold in pixels.
            config: Architecture settings for a newly built detector.
        """
        self.detector = detector or DetectorDescriptor(
            config,
            device=device,
            max_keypoints=max_keypoints,
        )
        self.ransac_threshold = ransac_threshold
        self.name = self.detector.name

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference crop.
            query: RGB query image.

        Returns:
            Keypoints, matches, and the estimated homography.
        """
        return self.match_detections(
            self.detector.detect(reference),
            self.detector.detect(query),
        )

    def match_detections(
        self,
        detections0: Detections,
        detections1: Detections,
    ) -> MatchResult:
        """Match two precomputed detection sets.

        Useful when the same image is queried many times: detect once per image,
        then match against every counterpart without re-running the backbone.

        Args:
            detections0: Detections for the reference.
            detections1: Detections for the query.

        Returns:
            Keypoints, matches, and the estimated homography.
        """
        pairs, scores = match_descriptors(
            detections0.descriptors,
            detections1.descriptors,
            ratio=self.detector.config.ratio,
        )
        source = (
            detections0.keypoints[[index0 for index0, _ in pairs]]
            if pairs
            else detections0.keypoints[:0]
        )
        target = (
            detections1.keypoints[[index1 for _, index1 in pairs]]
            if pairs
            else detections1.keypoints[:0]
        )
        homography, inlier_mask, inlier_count = estimate_homography(
            source,
            target,
            self.ransac_threshold,
        )
        return MatchResult(
            keypoints0=detections0.keypoints,
            keypoints1=detections1.keypoints,
            matches=pairs,
            scores=scores,
            keypoint_scores0=tuple(float(value) for value in detections0.scores),
            keypoint_scores1=tuple(float(value) for value in detections1.scores),
            homography=homography,
            inlier_mask=inlier_mask,
            match_count=len(pairs),
            inlier_count=inlier_count,
        )
