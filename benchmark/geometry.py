"""Box geometry and image loading helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from benchmark.matchers.base import MatchResult
from utils.dataset import Box, InstanceSample


def box_iou(first: Box, second: Box) -> float:
    """Compute the intersection-over-union of two boxes.

    Args:
        first: First ``(x1, y1, x2, y2)`` box.
        second: Second ``(x1, y1, x2, y2)`` box.

    Returns:
        The IoU in ``[0, 1]``.
    """
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def predicted_box(match: MatchResult, reference_shape: tuple[int, int]) -> Box | None:
    """Map the reference crop corners through the estimated homography.

    Args:
        match: Match result with an optional homography.
        reference_shape: ``(height, width)`` of the reference crop.

    Returns:
        The axis-aligned predicted query box, or ``None``.
    """
    if match.homography is None:
        return None
    height, width = reference_shape
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), match.homography
    ).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    if x2 <= x1 or y2 <= y1:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def reference_crop(sample: InstanceSample) -> np.ndarray:
    """Return the RGB reference crop for a sample.

    Args:
        sample: Reference/query pair.

    Returns:
        The cropped reference image.
    """
    image = sample.reference_image
    if image is None:
        image = load_rgb(sample.reference_path)
    if sample.reference_box is None:
        return image
    height, width = image.shape[:2]
    x1, y1, x2, y2 = sample.reference_box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return image[top:bottom, left:right]


def query_image(sample: InstanceSample) -> np.ndarray:
    """Return the RGB query image for a sample.

    Args:
        sample: Reference/query pair.

    Returns:
        The query image.
    """
    if sample.query_image is not None:
        return sample.query_image
    return load_rgb(sample.query_path)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB.

    Args:
        path: Image path.

    Returns:
        The RGB image.

    Raises:
        FileNotFoundError: If the image cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
