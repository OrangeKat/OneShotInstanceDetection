"""Keypoint selection and descriptor matching."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

__all__ = ["extract_keypoints", "match_descriptors"]


def extract_keypoints(
    boundary_logits: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick keypoints from a boundary map.

    A keypoint is a local maximum of the boundary probability, above a floor,
    kept in descending order until ``top_k``. Local maxima rather than random
    sampling is what keeps points on the outline instead of scattering them.

    Args:
        boundary_logits: ``(1, 1, h, w)`` logits.
        threshold: Minimum probability for a candidate.
        top_k: Maximum number of points.

    Returns:
        ``(N, 2)`` feature-cell coordinates and ``(N,)`` probabilities.
    """
    probability = torch.sigmoid(boundary_logits)[0, 0]
    pooled = F.max_pool2d(probability[None, None], 3, stride=1, padding=1)[0, 0]
    peaks = (probability >= pooled) & (probability >= threshold)
    indices = torch.nonzero(peaks, as_tuple=False)
    if indices.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    scores = probability[indices[:, 0], indices[:, 1]]
    order = torch.argsort(scores, descending=True)[:top_k]
    indices = indices[order]
    points = indices.flip(1).float().cpu().numpy()  # (x, y)
    return points, scores[order].detach().cpu().numpy()


def match_descriptors(
    descriptors0: np.ndarray,
    descriptors1: np.ndarray,
    *,
    ratio: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Match descriptors with mutual nearest neighbours and a ratio test.

    Args:
        descriptors0: ``(N, D)`` unit-length reference descriptors.
        descriptors1: ``(M, D)`` unit-length query descriptors.
        ratio: Maximum best/second-best similarity ratio.

    Returns:
        Matched index pairs and their similarities.
    """
    if len(descriptors0) == 0 or len(descriptors1) == 0:
        return (), ()
    similarity = descriptors0 @ descriptors1.T
    best1 = similarity.argmax(axis=1)
    best0 = similarity.argmax(axis=0)
    order = np.argsort(-similarity, axis=1)
    pairs: list[tuple[int, int]] = []
    scores: list[float] = []
    for index0, index1 in enumerate(best1):
        if best0[index1] != index0:
            continue
        best = float(similarity[index0, index1])
        if similarity.shape[1] > 1:
            second = float(similarity[index0, order[index0, 1]])
            if second > 0 and best / max(second, 1e-6) < 1.0 / ratio:
                continue
        pairs.append((index0, int(index1)))
        scores.append(best)
    return tuple(pairs), tuple(scores)
