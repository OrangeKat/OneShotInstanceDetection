"""The boundary-anchored model, exposed through the benchmark interface."""

from __future__ import annotations

import numpy as np
import torch

from osid.benchmark.matchers.base import MatchResult, estimate_homography
from osid.model.backbone import BackboneFeatures, Dinov3Backbone
from osid.model.config import ModelConfig
from osid.model.heads import BoundaryHead, DescriptorHead, FuseDecoder
from osid.model.matcher import extract_keypoints, match_descriptors

__all__ = ["BoundaryModel"]


class BoundaryModel:
    """Match a reference crop to a query with boundary-anchored keypoints.

    The pipeline is: frozen DINOv3 features → fused decoder at stride 4 →
    boundary and descriptor heads → local maxima on the boundary map as
    keypoints → mutual nearest-neighbour descriptor matching.

    Attributes:
        name: Benchmark identifier.
        config: Architecture settings.
        device: Torch device the heads live on.
    """

    name = "osid_boundary"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        device: str | None = None,
        max_keypoints: int | None = None,
        ransac_threshold: float | None = None,
    ) -> None:
        """Build the model.

        Args:
            config: Architecture settings; defaults are in :class:`ModelConfig`.
            device: ``cuda``, ``cpu``, or ``None`` for automatic selection.
            max_keypoints: Overrides ``config.top_k`` when given.
            ransac_threshold: Overrides ``config.ransac_threshold`` when given.
        """
        self.config = config or ModelConfig()
        if max_keypoints is not None:
            self.config = ModelConfig(
                **{**self.config.__dict__, "top_k": max_keypoints}
            )
        if ransac_threshold is not None:
            self.config = ModelConfig(
                **{**self.config.__dict__, "ransac_threshold": ransac_threshold}
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone = Dinov3Backbone(self.config, self.device)
        self.decoder = FuseDecoder(
            self.config.hidden_dim,
            self.config.out_dim,
        ).to(self.device)
        self.boundary = BoundaryHead(self.config.out_dim).to(self.device)
        self.descriptor = DescriptorHead(
            self.config.out_dim,
            self.config.descriptor_dim,
        ).to(self.device)

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        """Match a reference crop against a query image.

        Args:
            reference: RGB reference crop.
            query: RGB query image.

        Returns:
            Keypoints, matching descriptors, and the estimated homography.
        """
        reference_features = self.backbone(reference)
        query_features = self.backbone(query)
        reference_prediction = self._predict(reference_features)
        query_prediction = self._predict(query_features)

        points0, scores0, descriptors0 = reference_prediction
        points1, scores1, descriptors1 = query_prediction
        pairs, match_scores = match_descriptors(
            descriptors0,
            descriptors1,
            ratio=self.config.ratio,
        )
        source = points0[[index0 for index0, _ in pairs]] if pairs else points0[:0]
        target = points1[[index1 for _, index1 in pairs]] if pairs else points1[:0]
        homography, inlier_mask, inlier_count = estimate_homography(
            source,
            target,
            self.config.ransac_threshold,
        )
        return MatchResult(
            keypoints0=points0,
            keypoints1=points1,
            matches=pairs,
            scores=match_scores,
            keypoint_scores0=tuple(float(value) for value in scores0),
            keypoint_scores1=tuple(float(value) for value in scores1),
            homography=homography,
            inlier_mask=inlier_mask,
            match_count=len(pairs),
            inlier_count=inlier_count,
        )

    def _predict(
        self,
        features: BackboneFeatures,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the decoder and heads, then select keypoints.

        Args:
            features: Backbone features for one image.

        Returns:
            ``(N, 2)`` keypoints in original pixels, ``(N,)`` boundary scores,
            and ``(N, D)`` descriptors.
        """
        fused = self.decoder(features.layers)
        boundary_logits = self.boundary(fused)
        descriptor_map = self.descriptor(fused)
        points, scores = extract_keypoints(
            boundary_logits,
            threshold=self.config.boundary_threshold,
            top_k=self.config.top_k,
        )
        if len(points) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, self.config.descriptor_dim), dtype=np.float32),
            )
        descriptor_map = descriptor_map[0].detach().permute(1, 2, 0).cpu().numpy()
        rows = points[:, 1].astype(int)
        columns = points[:, 0].astype(int)
        descriptors = descriptor_map[rows, columns]
        return features.to_original(points), scores, descriptors
