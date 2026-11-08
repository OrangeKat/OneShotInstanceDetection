#!/usr/bin/env python3
"""Prepare the FlickrLogos-32 benchmark manifest.

The official FlickrLogos-32 download requires an email request. This script
instead pulls the FlickrLogos-32 positive images and boxes from the public
``Voxel51/OpenLogo`` mirror, where they are stored under numeric Flickr photo
IDs, and builds one reference/query split per logo class.

For each class, the image with the largest logo area becomes the reference
crop; the remaining images of that class become queries. Optional cross-class
images are added as negative queries.

Usage:
    uv run python scripts/prepare_flickrlogos32.py
    uv run python scripts/prepare_flickrlogos32.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/Voxel51/OpenLogo/resolve/main"
SAMPLES_URL = f"{BASE_URL}/samples.json"
EXPECTED_IMAGES = 2240
EXPECTED_CLASSES = 32
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


def main(argv: Sequence[str] | None = None) -> int:
    """Download FlickrLogos-32 images and write a benchmark manifest.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the FlickrLogos-32 benchmark manifest from OpenLogo."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/flickrlogos32"),
        help="Directory for images, metadata, and manifest.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        help="Cached OpenLogo samples.json. Downloads it when missing.",
    )
    parser.add_argument(
        "--negatives-per-class",
        type=int,
        default=10,
        help="Cross-class negative queries per logo class (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Negative sampling seed.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require images to exist locally.",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.samples or data_dir / "openlogo_samples.json"
    if not samples_path.exists():
        download_file(SAMPLES_URL, samples_path)

    payload = json.loads(samples_path.read_text())
    records = payload["samples"]
    images = filter_flickrlogos_images(records)
    if len(images) != EXPECTED_IMAGES:
        raise SystemExit(
            f"expected {EXPECTED_IMAGES} FlickrLogos-32 images, found {len(images)}"
        )
    classes = sorted({image.class_name for image in images})
    if len(classes) != EXPECTED_CLASSES:
        raise SystemExit(f"expected {EXPECTED_CLASSES} classes, found {len(classes)}")

    if args.skip_download:
        missing = [
            image for image in images if not image_path(data_dir, image).exists()
        ]
        if missing:
            raise SystemExit(
                f"{len(missing)} images are missing; run without --skip-download"
            )
    else:
        download_images(images, data_dir)

    manifest = build_manifest(
        images,
        negatives_per_class=args.negatives_per_class,
        seed=args.seed,
    )
    output = data_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest)} pairs across {len(classes)} classes to {output}")
    return 0


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    """Download one file with retries.

    Args:
        url: Source URL.
        destination: Destination path.
        retries: Number of attempts.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with destination.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            return
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def download_images(images: Sequence[LogoImage], data_dir: Path) -> None:
    """Download the selected images concurrently.

    Args:
        images: Images to download.
        data_dir: Dataset root directory.
    """
    pending = [image for image in images if not image_path(data_dir, image).exists()]
    if not pending:
        return
    print(f"downloading {len(pending)} images")
    failures: list[LogoImage] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                download_file,
                f"{BASE_URL}/{urllib.parse.quote(image.filepath, safe='/')}",
                image_path(data_dir, image),
            ): image
            for image in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            image = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(image)
                print(f"failed: {image.filepath}: {error}")
            if index % 100 == 0 or index == len(pending):
                print(f"\r  {index}/{len(pending)}", end="", flush=True)
    print()
    if failures:
        raise SystemExit(f"{len(failures)} image downloads failed")


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


def build_manifest(
    images: Sequence[LogoImage],
    *,
    negatives_per_class: int,
    seed: int,
) -> list[dict[str, object]]:
    """Build one reference and many queries per logo class.

    Args:
        images: FlickrLogos-32 positives.
        negatives_per_class: Cross-class negative queries per class.
        seed: Negative sampling seed.

    Returns:
        Manifest records.
    """
    by_class: dict[str, list[LogoImage]] = defaultdict(list)
    for image in images:
        by_class[image.class_name].append(image)

    records: list[dict[str, object]] = []
    for index, class_name in enumerate(sorted(by_class)):
        group = sorted(by_class[class_name], key=lambda image: image.name)
        by_area = sorted(
            group,
            key=lambda image: max(box_area(box) for box in image.boxes),
        )
        reference = by_area[len(by_area) // 2]
        reference_box = max(reference.boxes, key=box_area)
        for query in group:
            if query is reference:
                continue
            records.append(
                _record(
                    class_name,
                    reference,
                    reference_box,
                    query,
                    query.boxes,
                    negative=False,
                )
            )

        if negatives_per_class > 0:
            rng = random.Random(seed + index)
            candidates = [
                image
                for other, other_images in by_class.items()
                if other != class_name
                for image in other_images
            ]
            rng.shuffle(candidates)
            for negative in candidates[:negatives_per_class]:
                records.append(
                    _record(
                        class_name,
                        reference,
                        reference_box,
                        negative,
                        (),
                        negative=True,
                    )
                )
    return records


def _record(
    class_name: str,
    reference: LogoImage,
    reference_box: Box,
    query: LogoImage,
    query_boxes: Sequence[Box],
    *,
    negative: bool,
) -> dict[str, object]:
    """Create one manifest record.

    Args:
        class_name: Target logo class.
        reference: Reference image.
        reference_box: Reference crop.
        query: Query image.
        query_boxes: Target-class boxes in the query.
        negative: Whether the query is a negative example.

    Returns:
        A manifest record.
    """
    return {
        "dataset": "flickrlogos32",
        "instance_id": class_name,
        "category": "logo",
        "reference": f"images/{reference.class_name}/{reference.name}",
        "reference_box": [float(value) for value in reference_box],
        "query": f"images/{query.class_name}/{query.name}",
        "query_boxes": [[float(value) for value in box] for box in query_boxes],
        "metadata": {
            "class": class_name,
            "source": "Voxel51/OpenLogo (FlickrLogos-32 subset)",
            "negative": negative,
            "original": query.filepath,
        },
    }


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
        The box area.
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def image_path(data_dir: Path, image: LogoImage) -> Path:
    """Return the local path for an image.

    Args:
        data_dir: Dataset root directory.
        image: Image to locate.

    Returns:
        The local image path.
    """
    return data_dir / "images" / image.class_name / image.name


if __name__ == "__main__":
    raise SystemExit(main())
