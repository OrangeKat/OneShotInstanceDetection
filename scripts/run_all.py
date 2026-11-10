#!/usr/bin/env python3
"""Run every available dataset benchmark sequentially.

Each dataset has its own protocol: box AP for FlickrLogos-32, CUHK-SYSU, and
RPC; retrieval for Paris6k; native matcher validation for HPatches. Datasets
whose data or manifest is missing are skipped with a message.

Usage:
    uv run python scripts/run_all.py
    uv run python scripts/run_all.py --models sift orb akaze
    uv run python scripts/run_all.py --datasets flickrlogos32 paris6k
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dataset:
    """One benchmark entry.

    Attributes:
        name: Dataset name.
        protocol: ``box``, ``retrieval``, or ``hpatches``.
        required: Path that must exist to run the benchmark.
        output: Output directory.
    """

    name: str
    protocol: str
    required: Path
    output: Path


DATASETS = [
    Dataset(
        "flickrlogos32",
        "box",
        Path("data/flickrlogos32/manifest.json"),
        Path("flickrlogos32"),
    ),
    Dataset(
        "cuhk-sysu",
        "box",
        Path("data/cuhk-sysu/manifest.json"),
        Path("cuhk-sysu"),
    ),
    Dataset(
        "rpc",
        "box",
        Path("data/rpc/manifest.json"),
        Path("rpc"),
    ),
    Dataset(
        "paris6k",
        "retrieval",
        Path("data/paris6k/manifest.json"),
        Path("paris6k"),
    ),
    Dataset(
        "hpatches",
        "hpatches",
        Path("data/hpatches/hpatches-sequences-release"),
        Path("hpatches"),
    ),
]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected datasets in order.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run every available benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to pass to each benchmark.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Only run these dataset names.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit samples per dataset (smoke test).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for per-dataset outputs.",
    )
    args = parser.parse_args(argv)

    selected = [
        dataset
        for dataset in DATASETS
        if args.datasets is None or dataset.name in args.datasets
    ]
    failures: list[str] = []
    for dataset in selected:
        if not (_REPO_ROOT / dataset.required).exists():
            print(f"skip {dataset.name}: missing {dataset.required}", flush=True)
            continue
        command = [
            sys.executable,
            str(_REPO_ROOT / "main.py"),
            "--models",
            *args.models,
            "--output",
            str(_REPO_ROOT / args.output_root / dataset.output),
        ]
        if dataset.protocol == "box":
            command.extend(["--manifest", str(_REPO_ROOT / dataset.required)])
        elif dataset.protocol == "retrieval":
            command.extend(["--protocol", "retrieval", "--paris-dir", "data/paris6k"])
        else:
            command.extend(
                ["--protocol", "hpatches", "--hpatches-dir", "data/hpatches"]
            )
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        print(f"running {dataset.name} ({dataset.protocol})", flush=True)
        result = subprocess.run(command, cwd=_REPO_ROOT, check=False)
        if result.returncode != 0:
            failures.append(dataset.name)
            print(f"failed {dataset.name}", flush=True)

    if failures:
        raise SystemExit(f"failed datasets: {', '.join(failures)}")
    print("all selected datasets finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
