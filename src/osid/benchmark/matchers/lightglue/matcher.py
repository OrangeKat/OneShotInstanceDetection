"""Shared wrapper for LightGlue-based matchers."""

from __future__ import annotations

import numpy as np
import torch

from osid.benchmark.matchers.base import (
    FeatureMatcher,
    MatchResult,
    estimate_homography,
)
from osid.benchmark.matchers.lightglue._vendor import LightGlue


class LightGlueFeatureMatcher(FeatureMatcher):
    """Base class for a LightGlue extractor/matcher pair.

    Attributes:
        name: Benchmark name.
        features: LightGlue feature family (``superpoint`` or ``aliked``).
        extractor_cls: LightGlue extractor class.
    """

    name: str
    features: str
    extractor_cls: type[torch.nn.Module]

    def __init__(
        self,
        *,
        max_keypoints: int = 2048,
        ratio: float = 0.75,
        ransac_threshold: float = 3.0,
        device: str | None = None,
        flash: bool = False,
        filter_threshold: float = 0.1,
    ) -> None:
        """Initialize the extractor and LightGlue matcher.

        Args:
            max_keypoints: Keypoint budget per image.
            ratio: Accepted for interface compatibility; LightGlue uses
                ``filter_threshold`` instead.
            ransac_threshold: RANSAC reprojection threshold in pixels.
            device: ``cuda``, ``cpu``, or ``None`` for automatic selection.
            flash: Enable FlashAttention. Defaults to ``False`` for ROCm
                compatibility.
            filter_threshold: LightGlue match-score threshold.
        """
        super().__init__(
            max_keypoints=max_keypoints,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._extractor = self.extractor_cls(
            max_num_keypoints=max_keypoints,
        ).eval()
        self._extractor = self._extractor.to(self.device)
        self._matcher = LightGlue(
            features=self.features,
            flash=flash,
            filter_threshold=filter_threshold,
        ).eval()
        self._matcher = self._matcher.to(self.device)

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference image.
            query: RGB query image.

        Returns:
            Matches and the estimated reference-to-query homography.
        """
        with torch.inference_mode():
            features0 = self._extractor.extract(self._to_tensor(reference))
            features1 = self._extractor.extract(self._to_tensor(query))
            output = self._matcher({"image0": features0, "image1": features1})

            keypoints0 = features0["keypoints"][0].detach().cpu().numpy()
            keypoints1 = features1["keypoints"][0].detach().cpu().numpy()
            detector_scores0 = (
                features0["keypoint_scores"][0].detach().cpu().numpy()
                if "keypoint_scores" in features0
                else None
            )
            detector_scores1 = (
                features1["keypoint_scores"][0].detach().cpu().numpy()
                if "keypoint_scores" in features1
                else None
            )
            matched = output["matches"][0].detach().cpu().numpy()
            raw_scores = (
                output["scores"][0].detach().cpu().numpy()
                if "scores" in output
                else None
            )

        pairs = tuple((int(index0), int(index1)) for index0, index1 in matched)
        if not pairs:
            return MatchResult(
                keypoints0=keypoints0,
                keypoints1=keypoints1,
                keypoint_scores0=(
                    tuple(float(value) for value in detector_scores0)
                    if detector_scores0 is not None
                    else ()
                ),
                keypoint_scores1=(
                    tuple(float(value) for value in detector_scores1)
                    if detector_scores1 is not None
                    else ()
                ),
            )

        source = keypoints0[[index0 for index0, _ in pairs]]
        target = keypoints1[[index1 for _, index1 in pairs]]
        homography, inlier_mask, inlier_count = estimate_homography(
            source,
            target,
            self.ransac_threshold,
        )
        scores = (
            tuple(float(value) for value in raw_scores)
            if raw_scores is not None and len(raw_scores) == len(pairs)
            else ()
        )
        return MatchResult(
            keypoints0=keypoints0,
            keypoints1=keypoints1,
            matches=pairs,
            scores=scores,
            keypoint_scores0=(
                tuple(float(value) for value in detector_scores0)
                if detector_scores0 is not None
                else ()
            ),
            keypoint_scores1=(
                tuple(float(value) for value in detector_scores1)
                if detector_scores1 is not None
                else ()
            ),
            homography=homography,
            inlier_mask=inlier_mask,
            match_count=len(pairs),
            inlier_count=inlier_count,
        )

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert an RGB uint8 image to a batched float tensor.

        Args:
            image: RGB image.

        Returns:
            A ``(1, 3, H, W)`` float tensor in ``[0, 1]`` on the target device.
        """
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)
