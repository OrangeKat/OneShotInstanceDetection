"""Deterministic seeding for benchmark runs."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and OpenCV RANSAC.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
