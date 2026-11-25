#!/usr/bin/env python3
"""Classify benchmark failures into the taxonomy and write a report.

Reads one results file, labels every positive pair from its closest attempt,
and reports the cause distribution for the whole set and for the pairs no model
solves.

Usage:
    uv run python scripts/taxonomy.py
    uv run python scripts/taxonomy.py --results outputs/flickrlogos32-refs/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.data import load_rows  # noqa: E402
from src.analysis.tables import write_csv  # noqa: E402
from src.analysis.taxonomy import (  # noqa: E402
    CAUSE_NOTES,
    CAUSES,
    classify_row,
    pair_labels,
    summary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Classify failures by cause.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/flickrlogos32-refs/results.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/flickrlogos32-refs/taxonomy"),
    )
    return parser.parse_args(argv)


def render(
    records: Sequence[dict[str, object]],
    per_model: dict[str, Counter[str]],
    models: Sequence[str],
    total_hard: int,
) -> str:
    """Render the Markdown report.

    Args:
        records: Pair labels.
        per_model: Cause counts per model.
        models: Model names.
        total_hard: Number of pairs no model solves.

    Returns:
        Markdown document.
    """
    hard = [record for record in records if int(record["models_solved"]) == 0]
    overall = summary(records, hard_only=False)
    hard_summary = summary(records, hard_only=True)
    lines = [
        "# Failure taxonomy",
        "",
        "Every positive pair is labelled from the model that came closest "
        "(highest IoU), because that attempt got furthest and its failure mode "
        "is the informative one.",
        "",
        "| Cause | Definition |",
        "|---|---|",
    ]
    for cause in CAUSES:
        lines.append(f"| `{cause}` | {CAUSE_NOTES[cause]} |")

    lines.extend(
        [
            "",
            "## All positive pairs",
            "",
            f"{overall['total']} pairs.",
            "",
            "| Cause | Pairs | Share |",
            "|---|---:|---:|",
        ]
    )
    for cause in CAUSES:
        count = int(overall["causes"][cause])
        share = count / overall["total"] if overall["total"] else 0.0
        lines.append(f"| `{cause}` | {count} | {share:.1%} |")

    lines.extend(
        [
            "",
            "## Hard pairs (no model solves)",
            "",
            f"{total_hard} pairs.",
            "",
            "| Cause | Pairs | Share |",
            "|---|---:|---:|",
        ]
    )
    for cause in CAUSES:
        count = int(hard_summary["causes"][cause])
        share = count / total_hard if total_hard else 0.0
        lines.append(f"| `{cause}` | {count} | {share:.1%} |")

    details: Counter[str] = Counter(str(record["detail"]) for record in hard)
    lines.extend(
        [
            "",
            "## Hard pairs by detail",
            "",
            "| Detail | Pairs | Meaning |",
            "|---|---:|---|",
            f"| `off_target` | {details.get('off_target', 0)} | box landed off the instance (IoU < 0.1) |",
            f"| `no_homography` | {details.get('no_homography', 0)} | matches existed but no consistent geometry |",
            f"| `loose` | {details.get('loose', 0)} | box on the instance, IoU 0.1-0.3 |",
            f"| `near_miss` | {details.get('near_miss', 0)} | box on the instance, IoU 0.3-0.5 |",
            "",
            "Note: `no_signal` is empty at pair level because the dense matchers "
            "always return correspondences, so some model always clears the "
            "four-match floor. It is common per model — see the next table.",
            "",
            "## Cause by model (all failures)",
            "",
            "| Model | " + " | ".join(f"`{cause}`" for cause in CAUSES) + " |",
            "|---|" + "---:|" * len(CAUSES),
        ]
    )
    for model in models:
        counts = per_model.get(model, Counter())
        cells = " | ".join(str(counts.get(cause, 0)) for cause in CAUSES)
        lines.append(f"| {model} | {cells} |")

    lines.extend(
        [
            "",
            "## Hard pairs by class",
            "",
            "| Class | " + " | ".join(f"`{cause}`" for cause in CAUSES) + " |",
            "|---|" + "---:|" * len(CAUSES),
        ]
    )
    per_class = hard_summary["per_class"]
    for name in sorted(per_class, key=lambda key: -sum(per_class[key].values())):
        counts = per_class[name]
        cells = " | ".join(str(counts.get(cause, 0)) for cause in CAUSES)
        lines.append(f"| {name} | {cells} |")

    lines.extend(["", "## Samples", ""])
    for cause in ("no_signal", "ambiguous", "weak_geometry"):
        examples = [record for record in hard if record["cause"] == cause][:5]
        lines.append(f"**{cause}**")
        lines.append("")
        if not examples:
            lines.append("none")
        for record in examples:
            lines.append(
                f"- {record['class_name']} / {record['query']} — "
                f"best {record['best_model']} IoU {record['best_iou']}, "
                f"{record['matches']} matches, {record['inliers']} inliers, "
                f"coverage {record['coverage']}, stage `{record['stage']}`"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the taxonomy pass.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    payload = json.loads(args.results.read_text())
    models = [str(report["model"]) for report in payload["models"]]
    rows, _ = load_rows(args.results, args.manifest)
    records = pair_labels(rows, models)

    per_model: dict[str, Counter[str]] = {}
    for model in models:
        counts: Counter[str] = Counter()
        for row in rows:
            if row.model == model and not row.negative:
                counts[classify_row(row)] += 1
        per_model[model] = counts

    hard = [record for record in records if int(record["models_solved"]) == 0]
    hard_summary = summary(records, hard_only=True)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "pair_taxonomy.csv", records)
    (args.output / "taxonomy.md").write_text(
        render(records, per_model, models, len(hard))
    )

    print(f"{len(records)} positive pairs, {len(hard)} hard")
    for cause in CAUSES[:-1]:
        count = int(hard_summary["causes"][cause])
        share = count / len(hard) if hard else 0.0
        print(f"  {cause:14s} {count:4d}  {share:5.1%}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
