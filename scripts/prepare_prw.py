#!/usr/bin/env python3
"""Prepare a PRW person-search benchmark manifest.

PRW (Person Re-identification in the Wild) is a full-frame person-search
dataset: 11,816 frames, 932 identities, and 34,304 annotated boxes across six
cameras. Each query is a hand-drawn box crop; gallery frames carry the same
identity's boxes and camera IDs.

This script builds one reference/query pair per gallery frame containing the
query identity, plus sampled negative frames. It expects the extracted PRW
release, for example::

    data/prw/PRW-v16.04.20/
      frames/*.jpg
      annotations/*.jpg.mat
      query_box/*.jpg
      frame_test.mat
      frame_train.mat

The official package is distributed through Google Drive / Kaggle; see the
PRW project page. Use ``--skip-download`` once it is extracted.

Usage:
    uv run python scripts/prepare_prw.py
    uv run python scripts/prepare_prw.py --max-queries 200 --negatives 3
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

Box = tuple[float, float, float, float]
CANDIDATE_ROOTS = ("PRW-v16.04.20", "PRW", "")


@dataclass
class Query:
    """One PRW query crop.

    Attributes:
        name: Query image file name.
        identity: PRW identity id.
        camera: Camera id.
    """

    name: str
    identity: int
    camera: int


def main(argv: Sequence[str] | None = None) -> int:
    """Build the PRW test manifest.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Prepare the PRW manifest.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/prw"),
        help="Directory containing the extracted PRW release.",
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
        help="Maximum queries to include; 0 means all.",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=5,
        help="Negative gallery frames per query.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = parser.parse_args(argv)

    root = find_root(args.data_dir)
    frames = root / "frames"
    annotations = root / "annotations"
    queries_dir = root / "query_box"
    frame_list = root / "frame_test.mat"
    for path in (frames, annotations, queries_dir, frame_list):
        if not path.exists():
            raise SystemExit(
                f"missing PRW path: {path}\n"
                "Download the PRW package, extract it under data/prw, and keep "
                "the official frames/annotations/query_box layout."
            )

    gallery = load_gallery(annotations, frame_list)
    queries = load_queries(queries_dir)
    if not queries:
        raise SystemExit(f"no query crops found under {queries_dir}")

    by_identity: dict[int, list[tuple[str, Box]]] = {}
    frame_ids: dict[str, set[int]] = {}
    for frame, boxes in gallery.items():
        ids: set[int] = set()
        for identity, box in boxes:
            by_identity.setdefault(identity, []).append((frame, box))
            ids.add(identity)
        frame_ids[frame] = ids

    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []
    for query in queries[: args.max_queries or None]:
        positives = [
            (frame, box)
            for frame, box in by_identity.get(query.identity, [])
            if camera_of(frame) != query.camera
        ]
        positive_frames = {frame for frame, _ in positives}
        negative_pool = [
            frame
            for frame in gallery
            if frame not in positive_frames and query.identity not in frame_ids[frame]
        ]
        negatives = rng.sample(negative_pool, min(args.negatives, len(negative_pool)))
        for frame, box in positives:
            records.append(record(query, frame, (box,), args.data_dir, root))
        for frame in negatives:
            records.append(record(query, frame, (), args.data_dir, root))

    if not records:
        raise SystemExit("no PRW query/gallery pairs were produced")
    output = args.output or args.data_dir / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    positives = sum(bool(record_item["query_boxes"]) for record_item in records)
    print(
        f"wrote {len(records)} pairs ({positives} positive, "
        f"{len(records) - positives} negative) over "
        f"{len({record_item['instance_id'] for record_item in records})} queries "
        f"to {output}"
    )
    return 0


def find_root(data_dir: Path) -> Path:
    """Find the directory containing frames/annotations/query_box.

    Args:
        data_dir: User data directory.

    Returns:
        Dataset root.

    Raises:
        SystemExit: If no candidate layout exists.
    """
    for name in CANDIDATE_ROOTS:
        candidate = data_dir / name if name else data_dir
        if (candidate / "frames").is_dir() and (candidate / "annotations").is_dir():
            return candidate
    raise SystemExit(
        f"could not find frames/ and annotations/ under {data_dir}; "
        "expected PRW-v16.04.20 or the extracted root"
    )


def load_gallery(
    annotations: Path,
    frame_list: Path,
) -> dict[str, list[tuple[int, Box]]]:
    """Load test-frame boxes grouped by frame.

    Args:
        annotations: Annotation directory.
        frame_list: ``frame_test.mat`` path.

    Returns:
        Frame name to ``(identity, box)`` list.
    """
    names = mat_strings(frame_list)
    gallery: dict[str, list[tuple[int, Box]]] = {}
    for name in names:
        stem = Path(name).stem
        path = annotations / f"{stem}.jpg.mat"
        if not path.exists():
            path = annotations / f"{stem}.mat"
        boxes = annotation_boxes(path)
        if boxes:
            gallery[stem] = boxes
    return gallery


def load_queries(queries_dir: Path) -> list[Query]:
    """Load query crops and their identities/cameras.

    Args:
        queries_dir: Query crop directory.

    Returns:
        Queries with identity and camera.
    """
    queries: list[Query] = []
    for path in sorted(queries_dir.glob("*.jpg")):
        stem = path.stem
        if len(stem) < 6 or not stem[:3].isdigit():
            continue
        identity = int(stem[:3])
        camera = int(stem[5]) if stem[5].isdigit() else -1
        queries.append(Query(name=path.name, identity=identity, camera=camera))
    return queries


def annotation_boxes(path: Path) -> list[tuple[int, Box]]:
    """Read ``[id, x, y, w, h]`` rows from one annotation file.

    Args:
        path: MAT annotation path.

    Returns:
        Identity and absolute box pairs.
    """
    if not path.exists():
        return []
    data = loadmat(path)
    matrix = None
    for value in data.values():
        array = np.asarray(value)
        if (
            array.ndim == 2
            and array.shape[1] >= 5
            and np.issubdtype(array.dtype, np.number)
        ):
            matrix = array
            break
    if matrix is None:
        return []
    result: list[tuple[int, Box]] = []
    for row in matrix:
        identity = int(row[0])
        if identity <= 0:
            continue
        x, y, width, height = (float(value) for value in row[1:5])
        if width <= 0 or height <= 0:
            continue
        result.append((identity, (x, y, x + width, y + height)))
    return result


def mat_strings(path: Path) -> list[str]:
    """Read frame names from a MAT cell array.

    Args:
        path: MAT file path.

    Returns:
        Flat list of strings.
    """
    data = loadmat(path)
    names: list[str] = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.dtype.kind in {"U", "S"} and array.ndim == 2:
            names.extend("".join(row.astype(str)).strip() for row in array)
            continue
        for item in array.reshape(-1):
            names.append(_text(item))
    return [name for name in names if name and not name.startswith("__")]


def _text(value: object) -> str:
    """Unwrap a MAT cell into a string.

    Args:
        value: MAT cell value.

    Returns:
        String content.
    """
    array = np.asarray(value)
    while array.dtype == object and array.size == 1:
        array = np.asarray(array.reshape(-1)[0])
    flat = array.reshape(-1)
    return str(flat[0]) if flat.size else ""


def camera_of(frame: str) -> int:
    """Return the camera id encoded in a PRW frame name.

    Args:
        frame: Frame name such as ``c1s1_000001``.

    Returns:
        Camera id, or ``-1``.
    """
    return (
        int(frame[1])
        if len(frame) > 1 and frame[0] == "c" and frame[1].isdigit()
        else -1
    )


def record(
    query: Query,
    frame: str,
    query_boxes: Sequence[Box],
    data_dir: Path,
    root: Path,
) -> dict[str, object]:
    """Build one manifest record.

    Args:
        query: PRW query.
        frame: Gallery frame name.
        query_boxes: Boxes of the query identity in the gallery frame.
        data_dir: Manifest base directory.
        root: Dataset root.

    Returns:
        Manifest record.
    """
    reference = (root / "query_box" / query.name).relative_to(data_dir)
    gallery = (root / "frames" / f"{frame}.jpg").relative_to(data_dir)
    return {
        "dataset": "prw",
        "instance_id": f"prw-{Path(query.name).stem}",
        "category": "person",
        "reference": str(reference),
        "query": str(gallery),
        "query_boxes": [list(box) for box in query_boxes],
        "metadata": {
            "split": "test",
            "identity": query.identity,
            "camera": query.camera,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
