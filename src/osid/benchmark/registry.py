"""Model registry: baseline matcher classes."""

from __future__ import annotations

from osid.benchmark.matchers import (
    AKAZEMatcher,
    ALIKEDLightGlueMatcher,
    FlexGlueMatcher,
    LoFTRMatcher,
    ORBMatcher,
    RoMaMatcher,
    SIFTMatcher,
    SuperPointLightGlueMatcher,
)
from osid.benchmark.matchers.base import FeatureMatcher

MODELS: dict[str, type[FeatureMatcher]] = {
    "sift": SIFTMatcher,
    "orb": ORBMatcher,
    "akaze": AKAZEMatcher,
    "superpoint_lightglue": SuperPointLightGlueMatcher,
    "aliked_lightglue": ALIKEDLightGlueMatcher,
    "loftr": LoFTRMatcher,
    "roma": RoMaMatcher,
    "flexglue": FlexGlueMatcher,
}
