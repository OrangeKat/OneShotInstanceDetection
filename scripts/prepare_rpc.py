#!/usr/bin/env python3
"""Prepare an RPC category-level benchmark manifest.

RPC contains checkout scenes annotated with grocery product boxes and 200
product categories, but no cross-image instance identity. This script builds a
category-level one-shot manifest: a reference crop of a product category from
one checkout scene is matched against other checkout scenes, with every box of
that category in the query scene as ground truth.

The images and boxes come from the public Hugging Face mirror
``benjamintli/retail-product-checkout``, which repackages the official RPC
release as Parquet shards. The official release is CC BY-NC-SA 4.0.

Usage:
    uv run python scripts/prepare_rpc.py
    uv run python scripts/prepare_rpc.py --max-shards 2 --queries-per-category 3
    uv run python scripts/prepare_rpc.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

REPO = "benjamintli/retail-product-checkout"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/data"
SPLITS = {"test": 11, "train": 19, "validation": 3}
Box = tuple[float, float, float, float]


def main(argv: Sequence[str] | None = None) -> int:
    """Download RPC shards if needed and write a category-level manifest.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the RPC category-level one-shot manifest."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/rpc"),
        help="Directory for Parquet shards, scenes, and manifest.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="test",
        help="RPC split to use.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest path. Defaults to <data-dir>/manifest.json.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=1,
        help="Number of Parquet shards to use; 0 means the whole split.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Stop after this many scenes; 0 means no limit.",
    )
    parser.add_argument(
        "--queries-per-category",
        type=int,
        default=5,
        help="Maximum query scenes per product category.",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=2,
        help="Negative query scenes per category.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require cached Parquet shards instead of downloading them.",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    shards = shard_paths(data_dir, args.split)
    needed = shards if args.max_shards == 0 else shards[: args.max_shards]
    for shard in needed:
        if not shard.exists():
            if args.skip_download:
                raise SystemExit(f"missing Parquet shard: {shard}")
            download(shard_url(args.split, shard.name), shard)

    scenes = collect_scenes(needed, data_dir / "scenes" / args.split, args.max_scenes)
    if not scenes:
        raise SystemExit("no RPC scenes found in the selected shards")
    names = category_names(needed[0])
    records = build_manifest(
        scenes,
        names,
        QueryOptions(
            queries_per_category=args.queries_per_category,
            negatives=args.negatives,
            seed=args.seed,
            split=args.split,
        ),
    )
    if not records:
        raise SystemExit("no categories had two scenes with annotations")

    output = args.output or data_dir / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    categories = len({record["instance_id"] for record in records})
    positives = sum(bool(record["query_boxes"]) for record in records)
    print(
        f"wrote {len(records)} pairs ({positives} positive, "
        f"{len(records) - positives} negative) over {categories} categories "
        f"to {output}"
    )
    return 0


def shard_paths(data_dir: Path, split: str) -> list[Path]:
    """Return the expected Parquet shard paths for a split.

    Args:
        data_dir: RPC data directory.
        split: Split name.

    Returns:
        Shard paths in index order.
    """
    count = SPLITS[split]
    return [
        data_dir / "parquet" / f"{split}-{index:05d}-of-{count:05d}.parquet"
        for index in range(count)
    ]


def shard_url(split: str, name: str) -> str:
    """Build the Hugging Face URL for one Parquet shard.

    Args:
        split: Split name.
        name: Shard file name.

    Returns:
        Download URL.
    """
    return f"{BASE_URL}/{name}"


def category_names(shard: Path) -> list[str]:
    """Read the product category names embedded in a shard's schema.

    Args:
        shard: Parquet shard path.

    Returns:
        Category names indexed by category id.
    """
    metadata = pq.ParquetFile(shard).schema_arrow.metadata or {}
    payload = json.loads(metadata[b"huggingface"])
    features = payload["info"]["features"]
    names: list[str] = features["objects"]["category"]["feature"]["names"]
    return names


def download(url: str, destination: Path) -> None:
    """Download a Parquet shard with a small progress report.

    Args:
        url: Source URL.
        destination: Destination file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        with destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                seen += len(chunk)
                if total:
                    print(f"\r  {seen / total:6.1%}", end="", flush=True)
    print()


def collect_scenes(
    shards: Sequence[Path],
    output_dir: Path,
    max_scenes: int,
) -> dict[str, dict[int, list[Box]]]:
    """Decode scene images and collect boxes by category.

    Args:
        shards: Parquet shards to read.
        output_dir: Directory for extracted scene images.
        max_scenes: Optional scene limit.

    Returns:
        Scene id to ``category index -> boxes`` mapping.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes: dict[str, dict[int, list[Box]]] = {}
    total = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        batch_index = 0
        for batch in parquet.iter_batches(
            batch_size=64,
            columns=["image", "objects"],
        ):
            for row in batch.to_pylist():
                if max_scenes and total >= max_scenes:
                    return scenes
                image = decode_image(row)
                if image is None:
                    continue
                scene_id = f"{shard.stem}-{batch_index:06d}"
                batch_index += 1
                path = output_dir / f"{scene_id}.jpg"
                cv2.imwrite(str(path), image)
                boxes = boxes_by_category(row.get("objects"), image.shape)
                scenes[scene_id] = boxes
                total += 1
                if total % 250 == 0:
                    print(f"  decoded {total} scenes", flush=True)
    print(f"decoded {total} scenes")
    return scenes


def decode_image(row: Mapping[str, object]) -> np.ndarray | None:
    """Decode one Parquet image row.

    Args:
        row: Parquet row.

    Returns:
        BGR image, or ``None`` if the row has no image bytes.
    """
    image = row.get("image")
    if not isinstance(image, Mapping) or image.get("bytes") is None:
        return None
    raw = np.frombuffer(image["bytes"], dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def boxes_by_category(
    objects: object,
    shape: tuple[int, ...],
) -> dict[int, list[Box]]:
    """Group valid boxes by category index.

    Args:
        objects: Parquet ``objects`` value with ``bbox`` and ``category``.
        shape: Image shape.

    Returns:
        Category index to absolute ``[x1, y1, x2, y2]`` boxes.
    """
    if not isinstance(objects, Mapping):
        return {}
    height, width = shape[:2]
    grouped: dict[int, list[Box]] = defaultdict(list)
    for box, category in zip(objects.get("bbox", []), objects.get("category", [])):
        if len(box) != 4:
            continue
        x, y, box_width, box_height = (float(value) for value in box)
        x1, y1 = max(0.0, x), max(0.0, y)
        x2 = min(float(width), x + box_width)
        y2 = min(float(height), y + box_height)
        if x2 <= x1 or y2 <= y1:
            continue
        grouped[int(category)].append((x1, y1, x2, y2))
    return grouped


@dataclass
class QueryOptions:
    """Options for building the category manifest.

    Attributes:
        queries_per_category: Maximum positive query scenes per category.
        negatives: Negative query scenes per category.
        seed: Sampling seed.
        split: Split name used in metadata and paths.
    """

    queries_per_category: int
    negatives: int
    seed: int
    split: str


def build_manifest(
    scenes: Mapping[str, Mapping[int, list[Box]]],
    names: Sequence[str],
    options: QueryOptions,
) -> list[dict[str, object]]:
    """Build category-level reference/query pairs.

    Args:
        scenes: Scene id to ``category index -> boxes`` mapping.
        names: Category names indexed by category id.
        options: Query-building options.

    Returns:
        Manifest records.
    """
    by_category: dict[int, list[str]] = defaultdict(list)
    for scene_id, boxes in scenes.items():
        for category in boxes:
            by_category[category].append(scene_id)

    rng = random.Random(options.seed)
    scene_ids = sorted(scenes)
    records: list[dict[str, object]] = []
    for category in sorted(by_category):
        owners = sorted(by_category[category])
        if len(owners) < 2:
            continue
        reference = owners[0]
        reference_box = scenes[reference][category][0]
        queries = owners[1 : 1 + options.queries_per_category]
        negatives = [scene for scene in scene_ids if scene not in owners]
        sampled = rng.sample(negatives, min(options.negatives, len(negatives)))
        name = names[category] if category < len(names) else f"category-{category}"
        for query in queries:
            records.append(
                record(
                    name,
                    category,
                    reference,
                    reference_box,
                    query,
                    scenes[query][category],
                    options.split,
                )
            )
        for query in sampled:
            records.append(
                record(
                    name,
                    category,
                    reference,
                    reference_box,
                    query,
                    (),
                    options.split,
                )
            )
    return records


def record(
    instance_id: str,
    category: int,
    reference: str,
    reference_box: Box,
    query: str,
    query_boxes: Sequence[Box],
    split: str,
) -> dict[str, object]:
    """Build one manifest record.

    Args:
        instance_id: Category instance identifier.
        category: Category index.
        reference: Reference scene id.
        reference_box: Box of the category in the reference scene.
        query: Query scene id.
        query_boxes: Boxes of the category in the query scene.
        split: Split name.

    Returns:
        Manifest record.
    """
    return {
        "dataset": "rpc",
        "instance_id": instance_id,
        "category": instance_id,
        "reference": f"scenes/{split}/{reference}.jpg",
        "reference_box": list(reference_box),
        "query": f"scenes/{split}/{query}.jpg",
        "query_boxes": [list(box) for box in query_boxes],
        "metadata": {
            "split": split,
            "level": "category",
            "category_index": category,
            "reference_scene": reference,
            "query_scene": query,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
