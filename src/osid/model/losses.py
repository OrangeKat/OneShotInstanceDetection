"""The two losses the first training attempt needs.

Deliberately small: a boundary heatmap loss and a descriptor contrastive loss.
Anything else — coverage, subpixel offsets, curvature weighting — is a regulariser
or a refinement, and belongs after these two are shown to learn.

The heatmap target is built at the decoder's resolution rather than at full
resolution. A one-pixel boundary at 448 px is a quarter of a cell at stride 4,
so a full-resolution band target would be almost empty; a Gaussian at cell
resolution keeps the supervision soft and the target populated.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from torch.nn import functional as F

__all__ = [
    "boundary_target",
    "focal_loss",
    "infonce",
    "symmetry_mask",
]


def boundary_target(
    mask: np.ndarray,
    *,
    out_size: int = 112,
    sigma: float = 0.9,
) -> np.ndarray:
    """Rasterise a mask boundary into a Gaussian heatmap.

    Args:
        mask: Binary silhouette, ``(H, W)`` uint8 0/1.
        out_size: Side of the square target.
        sigma: Gaussian standard deviation in target cells.

    Returns:
        ``(1, out_size, out_size)`` float32 heatmap in ``[0, 1]``.
    """
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    heat = np.zeros((out_size, out_size), dtype=np.float32)
    if not contours:
        return heat[None]
    scale = out_size / float(mask.shape[1])
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
    points = points * scale
    radius = max(1, int(round(3.0 * sigma)))
    for x, y in points:
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - radius), min(out_size, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(out_size, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
        np.maximum(heat[y0:y1, x0:x1], patch.astype(np.float32), out=heat[y0:y1, x0:x1])
    return heat[None]


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Penalty-reduced focal loss for a keypoint heatmap.

    Args:
        logits: ``(B, 1, H, W)`` predicted logits.
        target: ``(B, 1, H, W)`` heatmap in ``[0, 1]``.
        alpha: Positive-class focusing exponent.
        beta: Negative-class down-weighting exponent.

    Returns:
        Scalar loss, normalised by the number of positive cells.
    """
    probability = torch.sigmoid(logits)
    positive = target >= 1.0
    negative = ~positive
    positive_loss = -((1.0 - probability) ** alpha) * torch.log(
        probability.clamp_min(1e-9)
    )
    negative_loss = (
        -((1.0 - target) ** beta)
        * (probability**alpha)
        * torch.log((1.0 - probability).clamp_min(1e-9))
    )
    positive_loss = torch.where(positive, positive_loss, torch.zeros_like(logits))
    negative_loss = torch.where(negative, negative_loss, torch.zeros_like(logits))
    positives = positive.sum().clamp_min(1)
    return (positive_loss.sum() + negative_loss.sum()) / positives


def infonce(
    descriptors0: torch.Tensor,
    descriptors1: torch.Tensor,
    *,
    temperature: float = 0.07,
    ignore: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE over corresponding descriptors.

    Args:
        descriptors0: ``(N, D)`` unit-length descriptors from view 0.
        descriptors1: ``(N, D)`` unit-length descriptors from view 1, aligned
            with ``descriptors0``.
        temperature: Softmax temperature.
        ignore: Optional ``(N, N)`` boolean mask; ``True`` entries are excluded
            from the denominator, which is how genuinely ambiguous pairs (a
            symmetric shape matching itself elsewhere) are kept from penalising
            a correct prediction.

    Returns:
        Scalar loss.
    """
    logits = descriptors0 @ descriptors1.T / temperature
    if ignore is not None:
        logits = logits.masked_fill(ignore, float("-inf"))
    labels = torch.arange(len(descriptors0), device=descriptors0.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def symmetry_mask(
    points: np.ndarray,
    *,
    threshold: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Flag pairs of points that are too close in object space to distinguish.

    Args:
        points: ``(N, 2)`` corresponding points in canonical object coordinates.
        threshold: Distance below which two points are treated as ambiguous.
        device: Device for the returned tensor.

    Returns:
        ``(N, N)`` boolean mask with ``True`` on ambiguous negatives.
    """
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    mask = distances < threshold
    np.fill_diagonal(mask, False)
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _sanity_checks() -> None:
    """Run numerical checks on the two losses."""
    rng = np.random.default_rng(0)
    mask = np.zeros((448, 448), dtype=np.uint8)
    cv2.circle(mask, (224, 224), 90, 1, thickness=-1)
    target = boundary_target(mask, out_size=112)
    rows, columns = np.nonzero(target[0] > 0.5)
    radius = np.hypot(columns * 4 - 224, rows * 4 - 224)
    print(
        f"target: {target.shape}, cells above 0.5: {len(rows)}, "
        f"distance to the true circle: mean {radius.mean():.1f} px"
    )

    logits = torch.full((1, 1, 112, 112), -6.0)
    logits[0, 0][torch.as_tensor(target[0] > 0.5)] = 6.0
    aligned = focal_loss(logits, torch.as_tensor(target)[None])
    wrong = focal_loss(-logits, torch.as_tensor(target)[None])
    print(f"focal: aligned {aligned.item():.4f} | inverted {wrong.item():.4f}")
    assert aligned < wrong

    anchors = F.normalize(
        torch.randn(32, 16, generator=torch.Generator().manual_seed(0)), dim=1
    )
    noise = F.normalize(anchors + 0.05 * torch.randn(32, 16), dim=1)
    matched = infonce(anchors, noise)
    shuffled = infonce(anchors, noise[torch.randperm(32)])
    print(f"infonce: matched {matched.item():.3f} | shuffled {shuffled.item():.3f}")
    assert matched < shuffled

    points = rng.normal(0.0, 0.1, (32, 2))
    ignore = symmetry_mask(points, threshold=0.15)
    masked = infonce(anchors, noise, ignore=ignore)
    print(
        f"infonce with symmetry mask: {masked.item():.3f} | ignored pairs: {int(ignore.sum())}"
    )
    assert masked <= matched + 1e-3


def main(argv: list[str] | None = None) -> int:
    """Run the loss checks.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Check the training losses.")
    parser.parse_args(argv)
    _sanity_checks()
    print("loss checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
