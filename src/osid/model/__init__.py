"""Boundary-anchored keypoint matching model.

Architecture, in reading order:

- ``config.py``   knobs (backbone, input size, layers, widths, thresholds)
- ``backbone.py`` frozen DINOv3 ViT, multi-layer features, pixel geometry
- ``heads.py``    fusion decoder at stride 4, boundary head, descriptor head
- ``matcher.py``  local-maxima keypoints, mutual-NN matching with a ratio test
- ``model.py``    the pipeline, exposed as ``match(reference, query)``
"""

from osid.model.config import ModelConfig
from osid.model.model import BoundaryModel

__all__ = ["BoundaryModel", "ModelConfig"]
