#!/usr/bin/env python3
"""Aggregate repeated benchmark runs (for example, different seeds).

Reads several ``results.json`` files for the same manifest and reports the
mean and standard deviation of hit rate, AP@0.5, and latency per model.

Usage:
    uv run python scripts/aggregate_runs.py \
        outputs/seed0/results.json outputs/seed1/results.json \
        --output outputs/seed_summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate repeated runs.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Aggregate repeated runs.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/aggregate"),
        help="Output path without extension.",
    )
    args = parser.parse_args(argv)

    per_model: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for path in args.runs:
        payload = json.loads(path.read_text())
        for report in payload["models"]:
            per_model[str(report["model"])].append(report)
    if not per_model:
        raise SystemExit("no model reports found")

    rows = []
    lines = [
        "# Repeated-run summary",
        "",
        f"- Runs: {len(args.runs)}",
        "",
        "| Model | Runs | Hit rate | AP@0.5 | p50 ms |",
        "|---|---:|---|---|---|",
    ]
    for model in sorted(per_model):
        reports = per_model[model]
        hits = [float(report["hit_rate"]) for report in reports]
        aps = [float(report["ap50"]) for report in reports]
        p50 = [float(report["latency_ms"]["p50"]) for report in reports]
        rows.append(
            {
                "model": model,
                "runs": len(reports),
                "hit_rate_mean": float(np.mean(hits)),
                "hit_rate_std": float(np.std(hits)),
                "ap50_mean": float(np.mean(aps)),
                "ap50_std": float(np.std(aps)),
                "p50_mean": float(np.mean(p50)),
            }
        )
        lines.append(
            f"| {model} | {len(reports)} "
            f"| {np.mean(hits):.3f} ± {np.std(hits):.3f} "
            f"| {np.mean(aps):.3f} ± {np.std(aps):.3f} "
            f"| {np.mean(p50):.1f} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(f"wrote {args.output.with_suffix('.md')}")
    return 0


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write rows as CSV.

    Args:
        path: Output path.
        rows: Rows to write.
    """
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
