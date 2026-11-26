"""Reference selection: candidate crops, selections, and defect notes.

FlickrLogos-32 ships many images per class, so the benchmark has to choose one
crop as the reference. That choice is a data decision, not a model one, and the
median-area rule sometimes lands on a crop that is occluded, low resolution, or
not the mark at all. This module lists the alternatives and stores the choice.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from utils.dataset import InstanceDataset
from utils.openlogo import Box, box_area, image_path, load_images

__all__ = [
    "Candidate",
    "SELECTION_PATH",
    "candidates_for",
    "crop_bytes",
    "is_selectable",
    "load_selection",
    "save_selection",
    "selection_for",
]

SELECTION_PATH = Path("configs/references.json")
STATUSES = ("ok", "low-res", "occluded", "wrong-content", "unusable")


@dataclass(frozen=True)
class Candidate:
    """One possible reference crop.

    Attributes:
        image: Image file name.
        box: Crop inside the image, or ``None`` for a pre-cropped reference.
        path: Local file path.
        width: Image width in pixels.
        height: Image height in pixels.
        is_current: Whether this is the reference the manifest currently uses.
    """

    image: str
    box: Box | None
    path: Path
    width: int
    height: int
    is_current: bool = False

    @property
    def crop_size(self) -> tuple[float, float]:
        """Return the crop width and height.

        Returns:
            Crop dimensions, falling back to the whole image without a box.
        """
        if self.box is None:
            return (float(self.width), float(self.height))
        return (self.box[2] - self.box[0], self.box[3] - self.box[1])


def candidates_for(
    dataset: InstanceDataset,
    instance: str,
    data_dir: Path,
) -> list[Candidate]:
    """Return the reference candidates for one instance.

    For FlickrLogos-32 the candidates are every class image with its largest
    box; for a pre-cropped reference there is a single candidate.

    Args:
        dataset: Manifest-backed dataset.
        instance: Instance identifier.
        data_dir: Dataset root directory.

    Returns:
        Candidate crops, largest first.
    """
    samples = [sample for sample in dataset.samples if sample.instance_id == instance]
    current = samples[0].reference_path.name if samples else ""
    current_box = samples[0].reference_box if samples else None

    samples_path = data_dir / "openlogo_samples.json"
    if samples_path.exists():
        images = [
            image for image in load_images(samples_path) if image.class_name == instance
        ]
        candidates = []
        for image in sorted(images, key=lambda item: item.name):
            box = max(image.boxes, key=box_area) if image.boxes else None
            candidates.append(
                Candidate(
                    image=image.name,
                    box=box,
                    path=image_path(data_dir, image),
                    width=image.width,
                    height=image.height,
                    is_current=image.name == current,
                )
            )
        return sorted(
            candidates,
            key=lambda candidate: candidate.crop_size[0] * candidate.crop_size[1],
            reverse=True,
        )

    if not samples:
        return []
    reference = samples[0].reference_path
    width, height = _image_size(reference)
    return [
        Candidate(
            image=reference.name,
            box=current_box,
            path=reference,
            width=width,
            height=height,
            is_current=True,
        )
    ]


def _image_size(path: Path) -> tuple[int, int]:
    """Return image dimensions without decoding pixels twice.

    Args:
        path: Image path.

    Returns:
        ``(width, height)``, or ``(0, 0)`` when the file cannot be read.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return (0, 0)
    height, width = image.shape[:2]
    return (width, height)


def is_selectable(
    dataset: InstanceDataset,
    data_dir: Path,
    *,
    probe: int = 5,
) -> bool:
    """Report whether a dataset offers a choice of reference crops.

    A dataset where every instance has a single reference cannot be reviewed
    as a picker, so it is left out of the References view.

    Args:
        dataset: Manifest-backed dataset.
        data_dir: Dataset root directory.
        probe: Instances to inspect before deciding.

    Returns:
        True when at least one probed instance has more than one candidate.
    """
    seen: set[str] = set()
    for sample in dataset.samples:
        if sample.instance_id in seen:
            continue
        seen.add(sample.instance_id)
        if len(candidates_for(dataset, sample.instance_id, data_dir)) > 1:
            return True
        if len(seen) >= probe:
            break
    return False


def crop_bytes(candidate: Candidate, *, height: int = 132) -> bytes:
    """Render one candidate crop as PNG bytes.

    Args:
        candidate: Candidate to render.
        height: Target display height in pixels.

    Returns:
        PNG-encoded crop, or an empty string when the file cannot be read.

    Raises:
        FileNotFoundError: If the image is missing.
    """
    image = cv2.imread(str(candidate.path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {candidate.path}")
    if candidate.box is not None:
        x1, y1, x2, y2 = (int(round(value)) for value in candidate.box)
        image = image[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)]
    source_height, source_width = image.shape[:2]
    if source_height == 0 or source_width == 0:
        raise ValueError("empty crop")
    scale = height / source_height
    width = max(1, int(round(source_width * scale)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".png", resized)
    if not ok:
        raise ValueError("image encoding failed")
    return buffer.tobytes()


def load_selection(path: Path = SELECTION_PATH) -> dict[str, dict[str, dict]]:
    """Read the stored reference selection.

    Args:
        path: Selection JSON path.

    Returns:
        Dataset to instance to record mapping, empty when the file is missing.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_selection(
    selection: Mapping[str, Mapping[str, Mapping[str, object]]],
    path: Path = SELECTION_PATH,
) -> None:
    """Write the reference selection.

    Args:
        selection: Dataset to instance to record mapping.
        path: Selection JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")


def selection_for(
    dataset_name: str,
    instance: str,
    selection: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object] | None:
    """Return the stored record for one instance.

    Args:
        dataset_name: Dataset key, e.g. ``flickrlogos32``.
        instance: Instance identifier.
        selection: Whole selection mapping.

    Returns:
        The stored record, or ``None``.
    """
    return selection.get(dataset_name, {}).get(instance)


def group_candidates(
    candidates: Sequence[Candidate],
) -> dict[str, list[Candidate]]:
    """Group candidates by nothing but keep a stable shape for the API.

    Args:
        candidates: Candidate crops.

    Returns:
        ``{"all": [...]}`` so the client can render uniformly across datasets.
    """
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    grouped["all"].extend(candidates)
    return dict(grouped)
