"""Feature fusion and the two prediction heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["BoundaryHead", "DescriptorHead", "FuseDecoder"]


class FuseDecoder(nn.Module):
    """Fuse several backbone layers and upsample to a quarter resolution.

    Boundary localisation needs more spatial detail than a 1/16 token grid, so
    each tapped layer is projected to a common width, summed, and upsampled
    twice. Output stride is 4 px.

    Attributes:
        in_dim: Channel count of the backbone layers; inferred on first call.
        out_dim: Channel count of the fused feature map.
    """

    def __init__(
        self, hidden_dim: int, out_dim: int, in_dim: int | None = None
    ) -> None:
        """Build the decoder.

        Args:
            hidden_dim: Width of the fused feature map.
            out_dim: Width handed to the heads.
            in_dim: Channel count of the backbone layers. Built eagerly when
                known so a saved state dict loads without a forward pass.
        """
        super().__init__()
        self.projections = (
            nn.ModuleList(nn.Conv2d(in_dim, hidden_dim, 1) for _ in range(4))
            if in_dim is not None
            else nn.ModuleList()
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, out_dim, 2, stride=2),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """Fuse layers and upsample.

        Args:
            layers: One ``(1, C, h, w)`` feature map per tapped block.

        Returns:
            ``(1, out_dim, h * 4, w * 4)`` fused features.
        """
        if not self.projections:
            in_channels = layers[0].shape[1]
            self.projections = nn.ModuleList(
                nn.Conv2d(in_channels, self.hidden_dim, 1) for _ in layers
            ).to(layers[0].device)
        target = layers[-1].shape[-2:]
        fused = None
        for projection, layer in zip(self.projections, layers):
            projected = projection(layer)
            if projected.shape[-2:] != target:
                projected = F.interpolate(
                    projected,
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                )
            fused = projected if fused is None else fused + projected
        assert fused is not None
        return self.up2(self.up1(self.fuse(fused)))


class BoundaryHead(nn.Module):
    """Predicts a per-pixel boundary logit."""

    def __init__(self, in_dim: int) -> None:
        """Build the head.

        Args:
            in_dim: Width of the fused feature map.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_dim, 1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict boundary logits.

        Args:
            features: ``(1, C, h, w)`` fused features.

        Returns:
            ``(1, 1, h, w)`` logits.
        """
        return self.conv(features)


class DescriptorHead(nn.Module):
    """Predicts an L2-normalised descriptor per feature cell."""

    def __init__(self, in_dim: int, descriptor_dim: int) -> None:
        """Build the head.

        Args:
            in_dim: Width of the fused feature map.
            descriptor_dim: Descriptor width.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_dim, descriptor_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict descriptors.

        Args:
            features: ``(1, C, h, w)`` fused features.

        Returns:
            ``(1, D, h, w)`` unit-length descriptors.
        """
        return F.normalize(self.conv(features), dim=1)
