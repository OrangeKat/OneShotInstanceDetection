"""Feature-matching baselines."""

from osid.benchmark.matchers.base import FeatureMatcher, MatchResult
from osid.benchmark.matchers.classical import (
    AKAZEMatcher,
    ORBMatcher,
    SIFTMatcher,
)
from osid.benchmark.matchers.flexglue import FlexGlueMatcher
from osid.benchmark.matchers.lightglue import (
    ALIKEDLightGlueMatcher,
    SuperPointLightGlueMatcher,
)
from osid.benchmark.matchers.loftr import LoFTRMatcher
from osid.benchmark.matchers.roma import RoMaMatcher

__all__ = [
    "AKAZEMatcher",
    "ALIKEDLightGlueMatcher",
    "FeatureMatcher",
    "FlexGlueMatcher",
    "LoFTRMatcher",
    "MatchResult",
    "ORBMatcher",
    "RoMaMatcher",
    "SIFTMatcher",
    "SuperPointLightGlueMatcher",
]
