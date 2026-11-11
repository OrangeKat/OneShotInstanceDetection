#!/usr/bin/env python3
"""Build a model x dataset comparison from benchmark result files.

Scans output directories for ``results.json`` and writes a Markdown plus CSV
matrix. The primary metric depends on the protocol: AP@0.5 for box datasets and
AUC@5 for HPatches.

Usage:
    uv run python scripts/compare_datasets.py
    uv run python scripts/compare_datasets.py --datasets flickrlogos32 hpatches
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

Metric = Callable[[Mapping[str, object]], object]
PRIMARY: dict[str, tuple[str, Metric]] = {
    "box": ("AP@0.5", lambda report: report.get("ap50")),
    "hpatches": ("AUC@5", lambda report: report.get("auc", {}).get("5")),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Scan result files and write the comparison.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Compare models across datasets.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Directory containing per-dataset result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/comparison"),
        help="Output path without extension.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Only include these result folder names.",
    )
    args = parser.parse_args(argv)

    datasets = collect(args.outputs)
    if args.datasets:
        wanted = set(args.datasets)
        datasets = [item for item in datasets if item[0] in wanted]
    if not datasets:
        raise SystemExit(f"no results.json found under {args.outputs}")

    models = sorted({model for _, _, _, reports in datasets for model in reports})
    rows: list[dict[str, object]] = []
    lines = [
        "# Model x dataset comparison",
        "",
        "| Model | " + " | ".join(item[0] for item in datasets) + " |",
        "|---|" + "---:|" * len(datasets),
    ]
    for model in models:
        cells = [
            _format(metric(reports.get(model, {})))
            for _, _, metric, reports in datasets
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
        rows.append(
            {
                "model": model,
                **{
                    name: metric(reports.get(model, {}))
                    for name, _, metric, reports in datasets
                },
            }
        )

    lines.extend(["", "## Metric per dataset", ""])
    for name, metric_name, _, _ in datasets:
        lines.append(f"- `{name}`: {metric_name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(f"wrote {args.output.with_suffix('.md')}")
    return 0


def collect(
    root: Path,
) -> list[tuple[str, str, Metric, dict[str, Mapping[str, object]]]]:
    """Collect per-dataset model metrics.

    Args:
        root: Output directory.

    Returns:
        ``(dataset, metric name, metric function, model reports)`` tuples.
    """
    datasets = []
    for path in sorted(root.glob("*/results.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        protocol = str(payload.get("config", {}).get("protocol", "box"))
        if protocol not in PRIMARY:
            continue
        metric_name, metric = PRIMARY[protocol]
        reports = {str(report["model"]): report for report in payload.get("models", [])}
        if reports:
            datasets.append((path.parent.name, metric_name, metric, reports))
    return datasets


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write comparison rows as CSV.

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
        lines.append(
            ",".join("" if row[field] is None else str(row[field]) for field in fields)
        )
    path.write_text("\n".join(lines) + "\n")


def _format(value: object) -> str:
    """Format an optional metric value.

    Args:
        value: Metric value or ``None``.

    Returns:
        Formatted string.
    """
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
