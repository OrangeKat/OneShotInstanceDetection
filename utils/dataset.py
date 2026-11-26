"""Manifest-backed datasets for one-shot instance detection.

A manifest is a JSON list (or a JSONL file) of reference/query pairs. Each
record describes one instance reference and one query image with its
ground-truth boxes::

    [
      {
        "dataset": "flickrlogos32",
        "instance_id": "adidas-1",
        "category": "logos",
        "reference": "refs/adidas-1.png",
        "reference_box": [0, 0, 200, 120],
        "query": "images/0001.jpg",
        "query_boxes": [[10, 20, 110, 90]],
        "metadata": {"split": "test"}
      }
    ]

Paths are relative to ``root`` (the manifest directory by default). Boxes use
absolute ``[x1, y1, x2, y2]`` pixel coordinates. ``reference_box`` is optional;
when omitted, the whole reference image is the instance crop.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

__all__ = ["Box", "InstanceDataset", "InstanceSample"]

Box = tuple[float, float, float, float]


@dataclass
class InstanceSample:
    """One reference/query pair with ground-truth boxes in the query image.

    Attributes:
        dataset: Dataset name, e.g. ``"flickrlogos32"``.
        instance_id: Instance identifier, unique within ``dataset``.
        reference_path: Path to the reference image.
        query_path: Path to the query image.
        query_boxes: Ground-truth boxes for the instance in the query image.
        reference_box: Optional crop of the instance inside the reference image.
        category: Optional category such as ``"logos"`` or ``"people"``.
        metadata: Free-form manifest metadata.
        reference_image: Loaded RGB reference image, or ``None``.
        query_image: Loaded RGB query image, or ``None``.
    """

    dataset: str
    instance_id: str
    reference_path: Path
    query_path: Path
    query_boxes: tuple[Box, ...]
    reference_box: Box | None = None
    category: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    reference_image: np.ndarray | None = None
    query_image: np.ndarray | None = None

    @property
    def pair_key(self) -> tuple[str, str, Path]:
        """Return the unique ``(dataset, instance, query)`` key."""
        return (self.dataset, self.instance_id, self.query_path)

    @property
    def reference_key(self) -> tuple[str, str]:
        """Return the unique ``(dataset, instance)`` key."""
        return (self.dataset, self.instance_id)


class InstanceDataset:
    """Load reference/query pairs from a JSON or JSONL manifest.

    Raises:
        FileNotFoundError: If the manifest or an image does not exist.
        ValueError: If a record is malformed or a pair is duplicated.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        root: str | Path | None = None,
        load_images: bool = True,
    ) -> None:
        """Initialize the dataset from a manifest.

        Args:
            manifest: Path to the manifest file.
            root: Base directory for relative paths. Defaults to the manifest
                directory.
            load_images: Load images when samples are accessed. Set to ``False``
                to inspect or filter metadata without decoding pixels.
        """
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else self.manifest.parent
        self.load_images = load_images
        records = _read_manifest(self.manifest)
        self.samples: tuple[InstanceSample, ...] = tuple(
            self._parse_record(record) for record in records
        )
        self._validate()

    def __len__(self) -> int:
        """Return the number of reference/query pairs."""
        return len(self.samples)

    def __getitem__(self, index: int) -> InstanceSample:
        """Return the sample at ``index``, loading images if enabled.

        Args:
            index: Index into manifest order.

        Returns:
            The requested sample.
        """
        sample = self.samples[index]
        return self.load(sample) if self.load_images else sample

    def __iter__(self) -> Iterator[InstanceSample]:
        """Iterate over reference/query pairs."""
        for index in range(len(self)):
            yield self[index]

    def load(self, sample: InstanceSample) -> InstanceSample:
        """Return ``sample`` with its reference and query images loaded.

        Args:
            sample: Metadata sample from this dataset.

        Returns:
            A copy of ``sample`` with RGB images attached.
        """
        return replace(
            sample,
            reference_image=_load_rgb(sample.reference_path),
            query_image=_load_rgb(sample.query_path),
        )

    def reference_crop(self, sample: InstanceSample) -> np.ndarray:
        """Return the reference instance as an RGB crop.

        Args:
            sample: Sample whose reference image and optional box are used.

        Returns:
            The cropped reference image. If ``reference_box`` is ``None``, the
            full reference image is returned.
        """
        image = (
            sample.reference_image
            if sample.reference_image is not None
            else _load_rgb(sample.reference_path)
        )
        if sample.reference_box is None:
            return image
        return _crop(image, sample.reference_box)

    def references(self) -> tuple[InstanceSample, ...]:
        """Return one metadata sample per unique reference instance."""
        seen: set[tuple[str, str]] = set()
        unique: list[InstanceSample] = []
        for sample in self.samples:
            if sample.reference_key not in seen:
                seen.add(sample.reference_key)
                unique.append(sample)
        return tuple(unique)

    def query_paths(self) -> tuple[Path, ...]:
        """Return unique query image paths in manifest order."""
        seen: set[Path] = set()
        unique: list[Path] = []
        for sample in self.samples:
            if sample.query_path not in seen:
                seen.add(sample.query_path)
                unique.append(sample.query_path)
        return tuple(unique)

    def _parse_record(self, record: Mapping[str, object]) -> InstanceSample:
        """Convert one manifest record into a sample.

        Args:
            record: Mapping read from the manifest.

        Returns:
            The parsed sample.

        Raises:
            ValueError: If required fields are missing or malformed.
        """
        required = ("dataset", "instance_id", "reference", "query", "query_boxes")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"manifest record missing keys: {missing}")

        raw_reference_box = record.get("reference_box")
        reference_box = (
            _parse_box(raw_reference_box) if raw_reference_box is not None else None
        )
        category = record.get("category")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")

        return InstanceSample(
            dataset=str(record["dataset"]),
            instance_id=str(record["instance_id"]),
            reference_path=self._resolve(record["reference"]),
            query_path=self._resolve(record["query"]),
            query_boxes=_parse_boxes(record["query_boxes"]),
            reference_box=reference_box,
            category=str(category) if category is not None else None,
            metadata=metadata,
        )

    def _resolve(self, value: object) -> Path:
        """Resolve a manifest path against ``root``.

        Args:
            value: Path string from the manifest.

        Returns:
            An absolute path.
        """
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def _validate(self) -> None:
        """Reject duplicate pairs and inconsistent reference definitions."""
        pairs: set[tuple[str, str, Path]] = set()
        references: dict[tuple[str, str], tuple[Path, Box | None]] = {}
        for sample in self.samples:
            if sample.pair_key in pairs:
                raise ValueError(f"duplicate pair: {sample.pair_key}")
            pairs.add(sample.pair_key)

            expected = (sample.reference_path, sample.reference_box)
            actual = references.setdefault(sample.reference_key, expected)
            if actual != expected:
                raise ValueError(
                    "inconsistent reference for "
                    f"{sample.reference_key}: {actual} != {expected}"
                )


def _read_manifest(path: Path) -> list[Mapping[str, object]]:
    """Read a JSON list or JSONL manifest.

    Args:
        path: Manifest path.

    Returns:
        The list of records.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If the payload is not a list of records.
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            records = payload.get("samples", payload)
        else:
            records = payload
    if not isinstance(records, list):
        raise ValueError("manifest must contain a list of records")
    return records


def _parse_boxes(value: object) -> tuple[Box, ...]:
    """Parse a list of boxes.

    Args:
        value: Raw ``query_boxes`` value.

    Returns:
        Parsed boxes.

    Raises:
        ValueError: If the value is not a list of valid boxes.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"query_boxes must be a list, got {type(value).__name__}")
    return tuple(_parse_box(box) for box in value)


def _parse_box(value: object) -> Box:
    """Parse one ``[x1, y1, x2, y2]`` box.

    Args:
        value: Raw box value.

    Returns:
        The parsed box.

    Raises:
        ValueError: If the box is malformed or has no area.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"box must have 4 coordinates, got {value!r}")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid box: {(x1, y1, x2, y2)}")
    return (x1, y1, x2, y2)


def _crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Crop ``image`` to ``box``, clipped to the image bounds.

    Args:
        image: RGB image.
        box: ``(x1, y1, x2, y2)`` crop box.

    Returns:
        The cropped image.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return image[top:bottom, left:right]


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB array.

    Args:
        path: Image path.

    Returns:
        An RGB image.

    Raises:
        FileNotFoundError: If the image cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
