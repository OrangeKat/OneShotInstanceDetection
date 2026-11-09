#!/usr/bin/env python3
"""Run one-factor sensitivity sweeps for the box benchmark.

Each configuration is a normal benchmark run; the script only changes one
setting at a time and collects hit rate, AP@0.5, and latency.

Usage:
    uv run python scripts/sweep.py --keypoints 512 1024 2048
    uv run python scripts/sweep.py --ransac 1 3 5 --models sift orb
    uv run python scripts/sweep.py --max-side 480 640 960 --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured sweeps.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run benchmark sensitivity sweeps.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
    )
    parser.add_argument("--models", nargs="+", default=["sift"])
    parser.add_argument("--max-samples", type=int, help="Limit samples per run.")
    parser.add_argument("--keypoints", nargs="+", type=int, help="Keypoint budgets.")
    parser.add_argument("--ransac", nargs="+", type=float, help="RANSAC thresholds.")
    parser.add_argument("--max-side", nargs="+", type=int, help="Max image sides.")
    parser.add_argument("--ratio", nargs="+", type=float, help="Lowe ratios.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sweeps/sweep.csv"),
        help="CSV output path.",
    )
    args = parser.parse_args(argv)

    dimensions = {
        "max_keypoints": args.keypoints,
        "ransac_threshold": args.ransac,
        "max_side": args.max_side,
        "ratio": args.ratio,
    }
    dimensions = {key: value for key, value in dimensions.items() if value}
    if not dimensions:
        raise SystemExit(
            "provide at least one of --keypoints/--ransac/--max-side/--ratio"
        )

    rows: list[dict[str, object]] = []
    for dimension, values in dimensions.items():
        for value in values:
            run_dir = args.output.parent / "_runs" / f"{dimension}-{value}"
            command = [
                sys.executable,
                str(_REPO_ROOT / "main.py"),
                "--manifest",
                str(args.manifest),
                "--models",
                *args.models,
                "--output",
                str(run_dir),
                f"--{dimension.replace('_', '-')}",
                str(value),
            ]
            if args.max_samples is not None:
                command.extend(["--max-samples", str(args.max_samples)])
            print(f"running {dimension}={value}", flush=True)
            subprocess.run(command, check=True, cwd=_REPO_ROOT)
            payload = json.loads((run_dir / "results.json").read_text())
            for report in payload["models"]:
                rows.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "model": report["model"],
                        "hit_rate": report["hit_rate"],
                        "ap50": report["ap50"],
                        "mean_iou": report["mean_iou"],
                        "p50_ms": report["latency_ms"]["p50"],
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"wrote {args.output}")
    return 0


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write sweep rows as CSV.

    Args:
        path: Output path.
        rows: Rows to write.
    """
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
