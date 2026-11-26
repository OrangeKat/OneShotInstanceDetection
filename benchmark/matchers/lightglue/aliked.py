"""ALIKED + LightGlue feature matching."""

from __future__ import annotations

from benchmark.matchers.lightglue._vendor import ALIKED
from benchmark.matchers.lightglue.matcher import LightGlueFeatureMatcher

__all__ = ["ALIKEDLightGlueMatcher"]


class ALIKEDLightGlueMatcher(LightGlueFeatureMatcher):
    """ALIKED detector/descriptor with LightGlue matching."""

    name = "aliked_lightglue"
    features = "aliked"
    extractor_cls = ALIKED
