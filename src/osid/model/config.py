"""Architecture settings for the boundary-anchored model."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelConfig"]


@dataclass(frozen=True)
class ModelConfig:
    """Everything that shapes the model.

    Attributes:
        backbone: timm name of the frozen feature extractor.
        pretrained: Load published weights instead of random initialisation.
        input_size: Long side the images are resized to before the backbone.
        layers: Transformer blocks tapped for features, in depth order.
        hidden_dim: Width of the fused multi-layer feature map.
        out_dim: Width handed to the two heads.
        descriptor_dim: Width of the per-pixel descriptor.
        top_k: Maximum keypoints kept per image.
        boundary_threshold: Minimum boundary probability for a keypoint.
        ratio: Maximum best/second-best descriptor similarity ratio.
        ransac_threshold: Reprojection threshold for the homography, in pixels.
    """

    backbone: str = "vit_small_patch16_dinov3"
    pretrained: bool = True
    input_size: int = 448
    layers: tuple[int, ...] = (3, 6, 9, 11)
    hidden_dim: int = 128
    out_dim: int = 128
    descriptor_dim: int = 128
    top_k: int = 512
    boundary_threshold: float = 0.3
    ratio: float = 0.9
    ransac_threshold: float = 3.0
