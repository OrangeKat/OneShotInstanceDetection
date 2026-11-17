"""Command-line entry point for the inspector UI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.pipeline import MODELS
from src.ui.server import serve

__all__ = ["parse_args", "main"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse inspector arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="main.py ui",
        description="Serve a local pair inspector.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
        help="Benchmark manifest to browse.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        choices=sorted(MODELS),
        help="Models offered in the UI.",
    )
    parser.add_argument(
        "--hard-csv",
        nargs="*",
        type=Path,
        default=None,
        help="Hard-pair CSVs; defaults to the analysis output for the dataset.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/ui/cache"),
        help="Directory for cached pair dumps.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Keypoint budget per image.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inspector.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    hard_csv = args.hard_csv
    if hard_csv is None:
        dataset = args.manifest.parent.name
        hard_csv = [Path("outputs") / dataset / "analysis" / "hard_pairs.csv"]
    return serve(
        args.manifest,
        models=args.models,
        hard_csv=hard_csv,
        cache_dir=args.cache,
        host=args.host,
        port=args.port,
        max_keypoints=args.max_keypoints,
    )
