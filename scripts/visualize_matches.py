#!/usr/bin/env python3
"""Render match overlays for selected benchmark pairs.

Unlike ``visualize_examples.py``, which shows only boxes, this script re-runs
the matcher and draws the correspondences: inlier lines green, outlier lines
red, GT boxes green, predictions red.

Usage:
    uv run python scripts/visualize_matches.py \
        --manifest data/prw/manifest-q200.json \
        --results outputs/prw-q200/results.json \
        --output outputs/prw-q200/matches \
        --models sift orb akaze
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline import MODELS, box_iou, predicted_box
from src.utils.dataset import InstanceDataset
from src.utils.diagnostics import Box, match_diagnostics
from src.utils.overlay import render_overlay

SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def main(argv: Sequence[str] | None = None) -> int:
    """Render the overlay gallery.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Render match overlays.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to render. Defaults to every model in the results.",
    )
    parser.add_argument("--good", type=int, default=2, help="Good pairs per model.")
    parser.add_argument("--bad", type=int, default=2, help="Bad pairs per model.")
    parser.add_argument(
        "--negatives",
        type=int,
        default=1,
        help="False-positive pairs per model.",
    )
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text())
    dataset = InstanceDataset(args.manifest, load_images=False)
    by_key = {
        (sample.instance_id, sample.query_path.name): index
        for index, sample in enumerate(dataset.samples)
    }
    reports = {str(report["model"]): report for report in payload["models"]}
    models = args.models or list(reports)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for model in models:
        report = reports.get(model)
        if report is None:
            continue
        matcher = MODELS[model](
            max_keypoints=args.max_keypoints,
            ransac_threshold=args.ransac_threshold,
        )
        model_dir = args.output / SAFE_NAME.sub("_", model)
        model_dir.mkdir(parents=True, exist_ok=True)
        selected = select_records(
            report["per_sample"],
            good=args.good,
            bad=args.bad,
            negatives=args.negatives,
        )
        for index, (kind, record) in enumerate(selected, start=1):
            key = (str(record["instance_id"]), Path(str(record["query"])).name)
            sample_index = by_key.get(key)
            if sample_index is None:
                continue
            sample = dataset.load(dataset.samples[sample_index])
            reference = dataset.reference_crop(sample)
            query = sample.query_image
            if query is None:
                continue
            result = matcher.match(reference, query)
            box = predicted_box(result, reference.shape[:2])
            iou = best_iou(box, sample.query_boxes)
            diagnostics = match_diagnostics(
                result,
                reference.shape[:2],
                query.shape[:2],
                box,
                sample.query_boxes,
                iou,
                0.5,
            )
            overlay = render_overlay(
                reference,
                query,
                result,
                box,
                sample.query_boxes,
                diagnostics,
                iou,
            )
            stem = (
                f"{index:02d}_{kind}_"
                f"{SAFE_NAME.sub('_', str(record['instance_id']))}_"
                f"{SAFE_NAME.sub('_', Path(str(record['query'])).stem)}"
            )
            image_path = model_dir / f"{stem}.jpg"
            cv2.imwrite(
                str(image_path),
                overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            relative = image_path.relative_to(args.output)
            rows.append(
                {
                    "model": model,
                    "kind": kind,
                    "instance": str(record["instance_id"]),
                    "query": Path(str(record["query"])).name,
                    "stage": str(diagnostics["stage"]),
                    "matches": str(result.match_count),
                    "inliers": str(result.inlier_count),
                    "iou": f"{iou:.3f}",
                    "image": str(relative),
                }
            )

    if not rows:
        raise SystemExit("no overlays were rendered")
    write_index(args.output / "index.md", rows)
    print(f"wrote {len(rows)} overlays to {args.output}")
    return 0


def select_records(
    records: Sequence[Mapping[str, object]],
    *,
    good: int,
    bad: int,
    negatives: int,
) -> list[tuple[str, Mapping[str, object]]]:
    """Select good, bad, and false-positive records.

    Args:
        records: Per-sample records for one model.
        good: Number of high-IoU examples.
        bad: Number of low-IoU examples.
        negatives: Number of false positives.

    Returns:
        ``(kind, record)`` pairs.
    """
    positives = [record for record in records if record["gt_boxes"]]
    negative_records = [record for record in records if not record["gt_boxes"]]
    good_records = sorted(
        positives,
        key=lambda record: float(record["best_iou"]),
        reverse=True,
    )[:good]
    used = {sample_key(record) for record in good_records}
    bad_records = [
        record
        for record in sorted(
            positives,
            key=lambda record: (
                float(record["best_iou"]),
                -int(record["inlier_count"]),
            ),
        )
        if sample_key(record) not in used
    ][:bad]
    used.update(sample_key(record) for record in bad_records)
    false_positives = [
        record
        for record in sorted(
            negative_records,
            key=lambda record: int(record["inlier_count"]),
            reverse=True,
        )
        if record["predicted_box"] is not None and sample_key(record) not in used
    ][:negatives]
    return (
        [("good", record) for record in good_records]
        + [("bad", record) for record in bad_records]
        + [("false_positive", record) for record in false_positives]
    )


def sample_key(record: Mapping[str, object]) -> tuple[str, str]:
    """Return a unique key for a per-sample record.

    Args:
        record: Per-sample record.

    Returns:
        ``(instance_id, query file name)``.
    """
    return (str(record["instance_id"]), Path(str(record["query"])).name)


def best_iou(box: Box | None, gt_boxes: Sequence[Box]) -> float:
    """Return the best IoU against the ground truth.

    Args:
        box: Predicted box or ``None``.
        gt_boxes: Ground-truth boxes.

    Returns:
        Best IoU, or ``0``.
    """
    if box is None or not gt_boxes:
        return 0.0
    return max(box_iou(box, gt) for gt in gt_boxes)


def write_index(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Write the Markdown gallery index.

    Args:
        path: Output path.
        rows: Overlay metadata.
    """
    lines = [
        "# Match overlays",
        "",
        "Green = RANSAC inliers, red = outliers; GT boxes green, predictions red.",
        "",
        "| Model | Kind | Instance | Query | Stage | Matches | Inliers | IoU |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['kind']} | {row['instance']} "
            f"| {row['query']} | {row['stage']} | {row['matches']} "
            f"| {row['inliers']} | {row['iou']} |"
        )
    lines.append("")
    for row in rows:
        lines.append(f"## {row['model']} — {row['kind']} — {row['query']}")
        lines.append("")
        lines.append(f"![{row['model']} {row['kind']}]({row['image']})")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
