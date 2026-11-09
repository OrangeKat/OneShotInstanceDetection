"""ALIKED + LightGlue feature matching."""

from __future__ import annotations

from src.models.feature_matching.lightglue._vendor import ALIKED
from src.models.feature_matching.lightglue.matcher import LightGlueFeatureMatcher

__all__ = ["ALIKEDLightGlueMatcher"]


class ALIKEDLightGlueMatcher(LightGlueFeatureMatcher):
    """ALIKED detector/descriptor with LightGlue matching."""

    name = "aliked_lightglue"
    features = "aliked"
    extractor_cls = ALIKED
