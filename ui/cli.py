"""Command-line entry point for the inspector UI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from benchmark.registry import MODELS
from ui.references import SELECTION_PATH
from ui.server import serve

__all__ = ["parse_args", "main"]

DEFAULT_MANIFESTS = {
    "flickrlogos32": Path("data/flickrlogos32/manifest.json"),
    "rpc": Path("data/rpc/manifest.json"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse inspector arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="main.py ui",
        description="Serve a local pair inspector and reference picker.",
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Datasets to expose, as name=path. Defaults to every dataset whose "
            "manifest exists under data/."
        ),
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
        help="Hard-pair CSVs; defaults to each dataset's analysis output.",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=SELECTION_PATH,
        help="Reference selection JSON.",
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


def _manifests(values: Sequence[str] | None) -> dict[str, Path]:
    """Resolve the datasets to expose.

    Args:
        values: ``name=path`` entries, or ``None`` for the defaults.

    Returns:
        Dataset key to manifest path.

    Raises:
        SystemExit: If no dataset is available.
    """
    if values:
        manifests: dict[str, Path] = {}
        for value in values:
            name, _, path = value.partition("=")
            if not path:
                name, path = Path(path or value).parent.name, value
            manifests[name] = Path(path)
        return manifests
    available = {
        name: path for name, path in DEFAULT_MANIFESTS.items() if path.exists()
    }
    if not available:
        raise SystemExit(
            "no manifest found; run make data-flickr or pass --manifests name=path"
        )
    return available


def _hard_csvs(
    manifests: dict[str, Path],
    values: Sequence[Path] | None,
) -> dict[str, list[Path]]:
    """Resolve hard-pair CSVs per dataset.

    Args:
        manifests: Dataset key to manifest path.
        values: Explicit CSV paths, or ``None`` for the analysis defaults.

    Returns:
        Dataset key to CSV paths.
    """
    if values:
        return {name: list(values) for name in manifests}
    return {
        name: [Path("outputs") / name / "analysis" / "hard_pairs.csv"]
        for name in manifests
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inspector.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    manifests = _manifests(args.manifests)
    return serve(
        manifests,
        models=args.models,
        hard_csv=_hard_csvs(manifests, args.hard_csv),
        cache_dir=args.cache,
        host=args.host,
        port=args.port,
        max_keypoints=args.max_keypoints,
        selection_path=args.selection,
    )
