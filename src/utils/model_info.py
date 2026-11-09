"""Model size helpers for benchmark reports."""

from __future__ import annotations

import torch

from src.models.feature_matching.base import FeatureMatcher

__all__ = ["parameter_count"]


def parameter_count(matcher: FeatureMatcher) -> int | None:
    """Count unique trainable parameters in a matcher.

    Args:
        matcher: Feature matcher.

    Returns:
        Parameter count, or ``None`` for matchers without torch modules.
    """
    total = 0
    seen: set[int] = set()
    for value in vars(matcher).values():
        if not isinstance(value, torch.nn.Module):
            continue
        for parameter in value.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total or None
