#!/usr/bin/env python3
"""Prepare a CUHK-SYSU person-search benchmark manifest.

CUHK-SYSU is distributed on request, so this script expects the extracted
dataset at ``--data-dir`` with the official layout::

    data/cuhk-sysu/
      Image/SSM/*.jpg
      annotation/Images.mat
      annotation/test/train_test/TestG50.mat

For each test query it builds one reference crop from the probe image and
queries from the gallery: positive gallery images carry the person's box,
negative images carry no box. Boxes in the MAT files are ``[x, y, w, h]`` and
are converted to ``[x1, y1, x2, y2]``.

The official dataset is available by request only:
    https://github.com/ShuangLI59/person_search

Usage:
    uv run python scripts/prepare_cuhk_sysu.py
    uv run python scripts/prepare_cuhk_sysu.py --max-queries 100 --negatives 2
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SPLIT_FILE = Path("annotation/test/train_test/TestG50.mat")
IMAGE_DIR = Path("Image/SSM")
Box = tuple[float, float, float, float]


@dataclass
class Query:
    """One test query and its gallery.

    Attributes:
        index: Query index in the MAT file.
        image: Probe image name.
        box: Probe ``[x, y, w, h]`` box.
        positives: Gallery images containing the person with their boxes.
        negatives: Gallery images without the person.
    """

    index: int
    image: str
    box: Box
    positives: list[tuple[str, Box]]
    negatives: list[str]


def main(argv: Sequence[str] | None = None) -> int:
    """Build the CUHK-SYSU test manifest.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the CUHK-SYSU person-search manifest."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cuhk-sysu"),
        help="Directory containing Image/ and annotation/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest path. Defaults to <data-dir>/manifest.json.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Maximum test queries to include; 0 means all.",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=2,
        help="Negative gallery images per query.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    split_path = data_dir / SPLIT_FILE
    images = data_dir / IMAGE_DIR
    if not split_path.exists() or not images.is_dir():
        raise SystemExit(
            f"missing CUHK-SYSU files under {data_dir}.\n"
            "Request the dataset from the authors, extract it, and keep the "
            "official layout:\n"
            f"  {images}\n  {split_path}"
        )

    queries = load_queries(split_path)
    if not queries:
        raise SystemExit(f"no test queries found in {split_path}")
    if args.max_queries:
        queries = queries[: args.max_queries]

    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []
    for query in queries:
        records.extend(build_records(query, args.negatives, rng))

    if not records:
        raise SystemExit("no usable query/gallery pairs found")

    output = args.output or data_dir / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    positives = sum(bool(record["query_boxes"]) for record in records)
    print(
        f"wrote {len(records)} pairs ({positives} positive, "
        f"{len(records) - positives} negative) over {len(queries)} queries "
        f"to {output}"
    )
    return 0


def load_queries(path: Path) -> list[Query]:
    """Read test queries from ``TestG50.mat``.

    Args:
        path: MAT file path.

    Returns:
        Parsed queries in file order.
    """
    variable = path.stem
    payload = loadmat(path)
    if variable not in payload:
        keys = [key for key in payload if not key.startswith("__")]
        if len(keys) != 1:
            raise SystemExit(f"could not find {variable} in {path}; keys={keys}")
        variable = keys[0]

    queries: list[Query] = []
    for index, item in enumerate(payload[variable].reshape(-1)):
        query_cell = item["Query"][0, 0]
        image = text_of(query_cell[0])
        box = box_from_value(query_cell[1])
        if box is None:
            continue
        positives: list[tuple[str, Box]] = []
        negatives: list[str] = []
        for entry in item["Gallery"].squeeze():
            gallery_image = text_of(entry[0])
            gallery_box = box_from_value(entry[1], allow_empty=True)
            if gallery_box is None:
                negatives.append(gallery_image)
            else:
                positives.append((gallery_image, gallery_box))
        if not positives:
            continue
        queries.append(
            Query(
                index=index,
                image=image,
                box=box,
                positives=positives,
                negatives=negatives,
            )
        )
    return queries


def text_of(value: object) -> str:
    """Unwrap a nested MAT value into a string.

    Args:
        value: Raw MAT value.

    Returns:
        The first string or text element.
    """
    array = np.asarray(value)
    while array.dtype == object:
        flat = array.reshape(-1)
        if flat.size == 0:
            return ""
        array = np.asarray(flat[0])
    flat = array.reshape(-1)
    return str(flat[0]) if flat.size else ""


def numeric_array(value: object) -> np.ndarray | None:
    """Unwrap nested MAT cells into a flat numeric array.

    Args:
        value: Raw MAT value.

    Returns:
        Flat float array, or ``None`` if no numeric data is present.
    """
    array = np.asarray(value)
    while array.dtype == object:
        flat = array.reshape(-1)
        if flat.size == 0:
            return None
        if flat.size == 1:
            array = np.asarray(flat[0])
            continue
        parts = [numeric_array(part) for part in flat]
        parts = [part for part in parts if part is not None and part.size]
        return np.concatenate(parts) if parts else None
    return array.astype(np.float64).reshape(-1)


def box_from_value(value: object, *, allow_empty: bool = False) -> Box | None:
    """Convert a MAT box from ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``.

    Args:
        value: Raw MAT value.
        allow_empty: Return ``None`` for empty boxes instead of failing.

    Returns:
        Converted box, or ``None`` when the box is empty or degenerate.

    Raises:
        ValueError: If a non-empty box is malformed.
    """
    coordinates = numeric_array(value)
    if coordinates is None or coordinates.size == 0:
        if allow_empty:
            return None
        raise ValueError("query box is empty")
    if coordinates.size < 4:
        if allow_empty:
            return None
        raise ValueError(f"box needs 4 values, got {coordinates.size}")
    x, y, width, height = coordinates[:4]
    if width <= 0 or height <= 0:
        if allow_empty:
            return None
        raise ValueError(f"degenerate box: {(x, y, width, height)}")
    return (float(x), float(y), float(x + width), float(y + height))


def build_records(
    query: Query,
    negatives: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    """Build manifest records for one query.

    Args:
        query: Parsed test query.
        negatives: Number of negative gallery images to sample.
        rng: Sampling seed.

    Returns:
        Manifest records.
    """
    instance = f"query-{query.index:04d}"
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for image, box in query.positives:
        if image in seen:
            continue
        seen.add(image)
        records.append(record(instance, query, image, (box,)))
    sampled = rng.sample(query.negatives, min(negatives, len(query.negatives)))
    for image in sampled:
        if image in seen:
            continue
        seen.add(image)
        records.append(record(instance, query, image, ()))
    return records


def record(
    instance: str,
    query: Query,
    image: str,
    query_boxes: Iterable[Box],
) -> dict[str, object]:
    """Build one manifest record.

    Args:
        instance: Instance identifier.
        query: Parsed test query.
        image: Gallery image name.
        query_boxes: Boxes of the person in the gallery image.

    Returns:
        Manifest record.
    """
    return {
        "dataset": "cuhk-sysu",
        "instance_id": instance,
        "category": "person",
        "reference": f"{IMAGE_DIR}/{query.image}",
        "reference_box": list(query.box),
        "query": f"{IMAGE_DIR}/{image}",
        "query_boxes": [list(box) for box in query_boxes],
        "metadata": {
            "split": "test",
            "query_index": query.index,
            "gallery_size": len(query.positives) + len(query.negatives),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
