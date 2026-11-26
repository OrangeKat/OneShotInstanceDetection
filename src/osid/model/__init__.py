"""Boundary-anchored keypoint detection and description.

The model is per-image: one image in, keypoints and descriptors out.

- ``config.py``    knobs (backbone, input size, layers, widths, thresholds)
- ``backbone.py``  frozen DINOv3 ViT, multi-layer features, pixel geometry
- ``heads.py``     fusion decoder at stride 4, boundary head, descriptor head
- ``detector.py``  the model: ``detect(image) -> Detections``
- ``matcher.py``   matching utilities (mutual NN, ratio test)
- ``adapter.py``   benchmark adapter: detector on both images plus a matcher
- ``hub.py``       save/load and Hugging Face upload of the trained heads
"""

from osid.model.adapter import BenchmarkAdapter
from osid.model.config import ModelConfig
from osid.model.detector import Detections, DetectorDescriptor

__all__ = ["BenchmarkAdapter", "Detections", "DetectorDescriptor", "ModelConfig"]
