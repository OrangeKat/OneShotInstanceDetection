"""Local visual inspector for benchmark pairs.

The package serves a single-page view over one manifest: pick a pair, pick a
matcher, and see the correspondences the matcher produced before RANSAC, with
each query point tied back to the reference keypoint it matched.
"""

from src.ui.server import serve

__all__ = ["serve"]
