#!/usr/bin/env python3
"""Prepare the Paris6k retrieval manifest.

Downloads the revisited Paris6k image archives and ground truth, then builds a
candidate-ranking manifest. Each query is a query-image ROI with the revisitop
easy and hard positives plus sampled negatives; junk images are excluded.

Paris6k is a retrieval benchmark: the database images have no detection boxes,
so the manifest is evaluated with the retrieval protocol (mAP, recall@k) rather
than box AP.

Usage:
    uv run python scripts/prepare_paris6k.py
    uv run python scripts/prepare_paris6k.py --negatives 50 --max-queries 10
    uv run python scripts/prepare_paris6k.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import tarfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

URLS = {
    "paris_1-v1.tgz": ("https://thor.robots.ox.ac.uk/parisbuildings/paris_1-v1.tgz"),
    "paris_2-v1.tgz": ("https://thor.robots.ox.ac.uk/parisbuildings/paris_2-v1.tgz"),
}
GROUND_TRUTH = "gnd_rparis6k.pkl"
GROUND_TRUTH_URL = (
    "http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl"
)
IMAGE_DIR = "images"


def main(argv: Sequence[str] | None = None) -> int:
    """Download Paris6k if needed and write a retrieval manifest.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the Paris6k retrieval manifest."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/paris6k"),
        help="Directory for archives, images, ground truth, and manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest path. Defaults to <data-dir>/manifest.json.",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=100,
        help="Negative candidates per query.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Maximum queries to include; 0 means all.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require already extracted images and ground truth.",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    images = data_dir / IMAGE_DIR
    ground_truth = data_dir / GROUND_TRUTH
    if not args.skip_download:
        for name, url in URLS.items():
            archive = data_dir / name
            if not archive.exists():
                download(url, archive)
        if not ground_truth.exists():
            download(GROUND_TRUTH_URL, ground_truth)
        if not images.exists():
            for name in URLS:
                extract(data_dir / name, images)

    if not images.is_dir() or not ground_truth.exists():
        raise SystemExit(
            f"missing Paris6k files under {data_dir}; run without --skip-download"
        )

    index = image_index(images)
    if not index:
        raise SystemExit(f"no images found under {images}")
    queries = build_queries(
        ground_truth,
        index,
        data_dir,
        negatives=args.negatives,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    if not queries:
        raise SystemExit("no Paris6k queries could be matched to images")

    output = args.output or data_dir / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queries, indent=2) + "\n")
    positives = sum(len(query["positives"]) for query in queries)
    candidates = sum(len(query["candidates"]) for query in queries)
    print(
        f"wrote {len(queries)} queries with {positives} positive and "
        f"{candidates - positives} negative candidates to {output}"
    )
    return 0


def download(url: str, destination: Path) -> None:
    """Download a file with a small progress report.

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


def extract(archive: Path, destination: Path) -> None:
    """Extract a tar archive.

    Args:
        archive: Archive path.
        destination: Extraction directory.
    """
    print(f"extracting {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination)


def image_index(root: Path) -> dict[str, Path]:
    """Index images by file stem.

    Args:
        root: Image directory.

    Returns:
        Stem to path mapping.
    """
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index[path.stem] = path
    return index


def build_queries(
    ground_truth: Path,
    index: Mapping[str, Path],
    data_dir: Path,
    *,
    negatives: int,
    max_queries: int,
    seed: int,
) -> list[dict[str, object]]:
    """Build retrieval queries from the revisitop ground truth.

    Args:
        ground_truth: ``gnd_rparis6k.pkl`` path.
        index: Image stem to path mapping.
        data_dir: Base directory for relative paths.
        negatives: Negative candidates per query.
        max_queries: Optional query limit.
        seed: Sampling seed.

    Returns:
        Manifest records.
    """
    with ground_truth.open("rb") as handle:
        payload = pickle.load(handle)
    imlist: list[str] = payload["imlist"]
    qimlist: list[str] = payload["qimlist"]
    gnd = payload["gnd"]
    rng = random.Random(seed)

    query_stems = [Path(name).stem for name in qimlist]
    database_stems = [Path(name).stem for name in imlist]
    queries: list[dict[str, object]] = []
    for index_query, query_stem in enumerate(query_stems):
        if max_queries and len(queries) >= max_queries:
            break
        query_path = index.get(query_stem)
        if query_path is None:
            continue
        annotation = gnd[index_query]
        box = annotation["bbx"]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            continue

        positive_stems = {
            database_stems[idx]
            for idx in list(annotation["easy"]) + list(annotation["hard"])
            if 0 <= idx < len(database_stems)
        }
        junk_stems = {
            database_stems[idx]
            for idx in annotation["junk"]
            if 0 <= idx < len(database_stems)
        }
        positives = [index[stem] for stem in sorted(positive_stems) if stem in index]
        if not positives:
            continue
        negative_pool = [
            stem
            for stem in database_stems
            if stem in index and stem not in positive_stems and stem not in junk_stems
        ]
        sampled = rng.sample(negative_pool, min(negatives, len(negative_pool)))
        negative_paths = [index[stem] for stem in sampled]
        candidates = list(dict.fromkeys(positives + negative_paths))
        queries.append(
            {
                "dataset": "paris6k",
                "query_id": index_query,
                "query": str(query_path.relative_to(data_dir)),
                "query_box": [float(value) for value in box],
                "positives": [str(path.relative_to(data_dir)) for path in positives],
                "candidates": [str(path.relative_to(data_dir)) for path in candidates],
                "metadata": {
                    "landmark": query_stem.split("_")[1] if "_" in query_stem else "",
                    "easy": len(annotation["easy"]),
                    "hard": len(annotation["hard"]),
                    "junk": len(annotation["junk"]),
                },
            }
        )
    return queries


if __name__ == "__main__":
    raise SystemExit(main())
