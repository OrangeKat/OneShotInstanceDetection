"""Per-image detector and descriptor.

One image in, keypoints and descriptors out. Matching lives elsewhere
(``adapter.py`` for the benchmark, or any matcher you want to compare), so the
detector can be trained, measured, and cached on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from osid.model.backbone import BackboneFeatures, Dinov3Backbone
from osid.model.config import ModelConfig
from osid.model.heads import BoundaryHead, DescriptorHead, FuseDecoder
from osid.model.matcher import extract_keypoints

__all__ = ["Detections", "DetectorDescriptor"]


@dataclass
class Detections:
    """What the model produces for one image.

    Attributes:
        keypoints: ``(N, 2)`` coordinates in the original image.
        scores: ``(N,)`` boundary probability per keypoint.
        descriptors: ``(N, D)`` unit-length descriptors.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray

    def __len__(self) -> int:
        """Return the number of keypoints.

        Returns:
            Keypoint count.
        """
        return int(len(self.keypoints))


class DetectorDescriptor:
    """Boundary-aware keypoint detector with descriptors.

    Frozen DINOv3 features → fusion decoder → boundary and descriptor heads →
    local maxima of the boundary map. No second image is involved, so the
    output can be cached per image and reused by any matcher.

    Attributes:
        name: Identifier used in reports.
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
    ) -> None:
        """Build the detector.

        Args:
            config: Architecture settings.
            device: ``cuda``, ``cpu``, or ``None`` for automatic selection.
            max_keypoints: Overrides ``config.top_k`` when given.
        """
        self.config = config or ModelConfig()
        if max_keypoints is not None:
            self.config = ModelConfig(
                **{**self.config.__dict__, "top_k": max_keypoints}
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone = Dinov3Backbone(self.config, self.device)
        self.decoder = FuseDecoder(self.config.hidden_dim, self.config.out_dim).to(
            self.device
        )
        self.boundary = BoundaryHead(self.config.out_dim).to(self.device)
        self.descriptor = DescriptorHead(
            self.config.out_dim,
            self.config.descriptor_dim,
        ).to(self.device)

    def detect(self, image: np.ndarray) -> Detections:
        """Detect keypoints and describe them.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.

        Returns:
            Keypoints, boundary scores, and descriptors.
        """
        features = self.backbone(image)
        return self._predict(features)

    def _predict(self, features: BackboneFeatures) -> Detections:
        """Run the heads and select keypoints.

        Args:
            features: Backbone features for one image.

        Returns:
            Detections in original image coordinates.
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
            return Detections(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                descriptors=np.zeros((0, self.config.descriptor_dim), dtype=np.float32),
            )
        grid = descriptor_map[0].detach().permute(1, 2, 0).cpu().numpy()
        rows = points[:, 1].astype(int)
        columns = points[:, 0].astype(int)
        return Detections(
            keypoints=features.to_original(points),
            scores=scores,
            descriptors=grid[rows, columns],
        )
