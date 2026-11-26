"""SuperPoint + LightGlue feature matching."""

from __future__ import annotations

from benchmark.matchers.lightglue._vendor import SuperPoint
from benchmark.matchers.lightglue.matcher import LightGlueFeatureMatcher

__all__ = ["SuperPointLightGlueMatcher"]


class SuperPointLightGlueMatcher(LightGlueFeatureMatcher):
    """SuperPoint detector/descriptor with LightGlue matching."""

    name = "superpoint_lightglue"
    features = "superpoint"
    extractor_cls = SuperPoint
