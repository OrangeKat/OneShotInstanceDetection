#!/usr/bin/env python3
"""Prepare HPatches sequences and a box-AP manifest.

Downloads and extracts the official HPatches sequence release when needed, then
writes a manifest pairing each sequence reference image with its five target
images. The native matcher validation lives in ``validate_hpatches.py``.

Usage:
    uv run python scripts/prepare_hpatches.py
    uv run python scripts/prepare_hpatches.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

URL = (
    "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/"
    "hpatches-sequences-release.zip"
)
ARCHIVE_NAME = "hpatches-sequences-release.zip"
SEQUENCE_DIR = "hpatches-sequences-release"
TARGETS = range(2, 7)


def main(argv: Sequence[str] | None = None) -> int:
    """Download HPatches if needed and write the box-AP manifest.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the HPatches sequences and manifest."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/hpatches"),
        help="Directory for the archive, sequences, and manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest path. Defaults to <data-dir>/manifest.json.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require an existing archive or extracted sequence directory.",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    archive = data_dir / ARCHIVE_NAME
    sequences = data_dir / SEQUENCE_DIR
    if not sequences.exists():
        if not archive.exists():
            if args.skip_download:
                raise SystemExit(f"missing HPatches archive: {archive}")
            download(URL, archive)
        extract(archive, data_dir)

    output = args.output or data_dir / "manifest.json"
    records = build_manifest(sequences, data_dir)
    if not records:
        raise SystemExit(f"no HPatches sequences found under {sequences}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {len(records)} reference/query pairs to {output}")
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


def extract(archive: Path, data_dir: Path) -> None:
    """Extract the HPatches zip archive.

    Args:
        archive: Archive path.
        data_dir: Extraction directory.
    """
    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(data_dir)


def build_manifest(sequences_dir: Path, root: Path) -> list[dict[str, object]]:
    """Build manifest records from extracted HPatches sequences.

    Args:
        sequences_dir: Directory containing the sequence folders.
        root: Base directory used to relativise image paths.

    Returns:
        Manifest records in dataset order.
    """
    records: list[dict[str, object]] = []
    for sequence_dir in sorted(sequences_dir.iterdir()):
        if not sequence_dir.is_dir():
            continue
        reference = sequence_dir / "1.ppm"
        image = cv2.imread(str(reference), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        split = _split_name(sequence_dir.name)
        for target in TARGETS:
            query = sequence_dir / f"{target}.ppm"
            homography_path = sequence_dir / f"H_1_{target}"
            if not query.exists() or not homography_path.exists():
                continue
            homography = np.loadtxt(homography_path)
            if homography.shape != (3, 3):
                continue
            box = _projected_box(homography, width, height)
            if box is None:
                continue
            records.append(
                {
                    "dataset": "hpatches",
                    "instance_id": sequence_dir.name,
                    "category": split,
                    "reference": _relative(reference, root),
                    "query": _relative(query, root),
                    "query_boxes": [list(box)],
                    "metadata": {
                        "sequence": sequence_dir.name,
                        "target": target,
                        "split": split,
                        "homography": [float(value) for value in homography.ravel()],
                    },
                }
            )
    return records


def _split_name(sequence: str) -> str:
    """Return the HPatches split for a sequence name.

    Args:
        sequence: Sequence folder name.

    Returns:
        Split name.
    """
    if sequence.startswith("i_"):
        return "illumination"
    if sequence.startswith("v_"):
        return "viewpoint"
    return "unknown"


def _projected_box(
    homography: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Map reference-image corners through a homography.

    Args:
        homography: Reference-to-query homography.
        width: Reference image width.
        height: Reference image height.

    Returns:
        An axis-aligned query box, or ``None`` for a degenerate projection.
    """
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    projected = cv2.perspectiveTransform(corners.reshape(1, -1, 2), homography).reshape(
        -1, 2
    )
    if not np.isfinite(projected).all():
        return None
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    if x2 <= x1 or y2 <= y1:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def _relative(path: Path, root: Path) -> str:
    """Return a POSIX path relative to ``root``.

    Args:
        path: Target path.
        root: Base directory.

    Returns:
        Relative path as a string.
    """
    return path.relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
