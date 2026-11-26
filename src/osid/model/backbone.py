"""Frozen DINOv3 feature extractor."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import timm
import torch
from torch import nn

from osid.model.config import ModelConfig

__all__ = ["BackboneFeatures", "Dinov3Backbone"]


@dataclass
class BackboneFeatures:
    """Feature maps plus the geometry needed to map them back to pixels.

    Attributes:
        layers: One ``(1, C, h, w)`` map per tapped block, all at the same stride.
        stride: Pixels per feature cell in the resized image.
        resized: ``(width, height)`` of the image after resizing.
        original: ``(width, height)`` of the image before resizing.
    """

    layers: list[torch.Tensor]
    stride: int
    resized: tuple[int, int]
    original: tuple[int, int]

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Map feature-cell coordinates to original image pixels.

        Args:
            points: ``(N, 2)`` coordinates in feature cells.

        Returns:
            ``(N, 2)`` coordinates in the original image.
        """
        scale_x = self.original[0] / (self.resized[0] / self.stride)
        scale_y = self.original[1] / (self.resized[1] / self.stride)
        return points * np.array([scale_x, scale_y], dtype=np.float32)


class Dinov3Backbone(nn.Module):
    """DINOv3 ViT, frozen, exposing multi-layer feature maps.

    The backbone is never trained here: only the decoder and the heads learn.
    That keeps the memory budget small and the comparison honest, since the
    features are the published ones.

    Attributes:
        config: Architecture settings.
        model: The timm model.
        stride: Pixels per feature cell.
    """

    def __init__(self, config: ModelConfig, device: torch.device | None = None) -> None:
        """Load the backbone and freeze it.

        Args:
            config: Architecture settings.
            device: Device to place the backbone on.
        """
        super().__init__()
        self.config = config
        self.model = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        if device is not None:
            self.model.to(device)
        self.stride = int(self.model.patch_embed.patch_size[0])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, image: np.ndarray) -> BackboneFeatures:
        """Extract multi-layer features from an RGB image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.

        Returns:
            Feature maps and their geometry.
        """
        tensor, resized = self._prepare(image)
        with torch.no_grad():
            layers = self.model.forward_intermediates(
                tensor,
                indices=list(self.config.layers),
                intermediates_only=True,
            )
        return BackboneFeatures(
            layers=list(layers),
            stride=self.stride,
            resized=resized,
            original=(image.shape[1], image.shape[0]),
        )

    def _prepare(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Resize, pad to the patch grid, and normalise an image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.

        Returns:
            A batched tensor and the resized ``(width, height)``.
        """
        height, width = image.shape[:2]
        longest = max(height, width)
        scale = self.config.input_size / longest
        new_width = max(
            self.stride, int(round(width * scale / self.stride)) * self.stride
        )
        new_height = max(
            self.stride, int(round(height * scale / self.stride)) * self.stride
        )
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        device = next(self.model.parameters()).device
        tensor = (
            torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
            / 255.0
        )
        mean = self.mean.to(device)
        std = self.std.to(device)
        return (tensor - mean) / std, (new_width, new_height)
