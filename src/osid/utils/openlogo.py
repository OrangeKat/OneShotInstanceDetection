"""OpenLogo / FlickrLogos-32 annotation loading.

The official FlickrLogos-32 download needs an email request, so the benchmark
uses the public ``Voxel51/OpenLogo`` mirror. This module turns its sample
records into class-labelled images with absolute boxes, and is shared by the
dataset preparation script and the reference picker in the UI.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Box",
    "LogoImage",
    "absolute_box",
    "box_area",
    "filter_flickrlogos_images",
    "image_path",
    "load_images",
]

NUMERIC_NAME = re.compile(r"^\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
TEXT_CLASSES = {"adidas_text": "adidas", "heineken_text": "heineken"}
Box = tuple[float, float, float, float]


@dataclass
class LogoImage:
    """One FlickrLogos-32 image with absolute boxes for one class.

    Attributes:
        filepath: Path inside the OpenLogo repository.
        name: Image file name.
        class_name: Logo class after text-class remapping.
        width: Image width.
        height: Image height.
        boxes: Absolute ``(x1, y1, x2, y2)`` boxes for the class.
    """

    filepath: str
    name: str
    class_name: str
    width: int
    height: int
    boxes: tuple[Box, ...]


def load_images(samples_path: Path) -> list[LogoImage]:
    """Load FlickrLogos-32 images from a cached OpenLogo samples file.

    Args:
        samples_path: Path to ``openlogo_samples.json``.

    Returns:
        Images with mapped class labels and absolute boxes.
    """
    payload = json.loads(samples_path.read_text())
    return filter_flickrlogos_images(payload["samples"])


def filter_flickrlogos_images(
    records: Sequence[Mapping[str, object]],
) -> list[LogoImage]:
    """Extract FlickrLogos-32 positives from OpenLogo sample records.

    Args:
        records: OpenLogo ``samples`` records.

    Returns:
        Images with mapped class labels and absolute boxes.
    """
    images: list[LogoImage] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        filepath = str(record["filepath"])
        name = Path(filepath).name
        if not NUMERIC_NAME.match(name):
            continue
        width = int(record["width"])
        height = int(record["height"])
        ground_truth = record["ground_truth"]
        detections = ground_truth.get("detections", [])
        boxes_by_class: dict[str, list[Box]] = defaultdict(list)
        for detection in detections:
            label = TEXT_CLASSES.get(str(detection["label"]), str(detection["label"]))
            box = absolute_box(detection, width, height)
            if box is not None:
                boxes_by_class[label].append(box)
        if len(boxes_by_class) != 1:
            continue
        class_name, boxes = next(iter(boxes_by_class.items()))
        if (class_name, name) in seen:
            continue
        seen.add((class_name, name))
        images.append(
            LogoImage(
                filepath=filepath,
                name=name,
                class_name=class_name,
                width=width,
                height=height,
                boxes=tuple(boxes),
            )
        )
    return images


def absolute_box(
    detection: Mapping[str, object],
    width: int,
    height: int,
) -> Box | None:
    """Convert a normalized FiftyOne detection to an absolute box.

    Args:
        detection: Detection with a normalized ``bounding_box``.
        width: Image width.
        height: Image height.

    Returns:
        A clipped absolute box, or ``None`` for a degenerate box.
    """
    x, y, box_width, box_height = (float(value) for value in detection["bounding_box"])
    x1 = max(0.0, min(width - 1.0, x * width))
    y1 = max(0.0, min(height - 1.0, y * height))
    x2 = max(0.0, min(width - 1.0, (x + box_width) * width))
    y2 = max(0.0, min(height - 1.0, (y + box_height) * height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def box_area(box: Box) -> float:
    """Return the area of a box.

    Args:
        box: ``(x1, y1, x2, y2)`` box.

    Returns:
        Width times height.
    """
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def image_path(data_dir: Path, image: LogoImage) -> Path:
    """Return the local path for an image.

    Args:
        data_dir: Dataset root directory.
        image: Image to locate.

    Returns:
        Path inside ``data_dir``.
    """
    return data_dir / "images" / image.class_name / image.name
