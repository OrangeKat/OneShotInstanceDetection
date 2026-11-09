#!/usr/bin/env python3
"""Save annotated good and bad prediction examples from a benchmark run.

The script reads the per-sample records written by ``src/pipeline.py`` and
renders the reference crop next to the query image with ground-truth boxes in
green and predicted boxes in red. It saves a mix of successful predictions,
failures, and false positives on negative queries.

Usage:
    uv run python scripts/visualize_examples.py
    uv run python scripts/visualize_examples.py --good 3 --bad 3
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

Box = tuple[float, float, float, float]
SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def main(argv: Sequence[str] | None = None) -> int:
    """Render example predictions from a benchmark result file.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Visualize benchmark examples.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
        help="Benchmark manifest.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/flickrlogos32/results.json"),
        help="Results JSON from src/pipeline.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/flickrlogos32/examples"),
        help="Directory for rendered examples.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Only render these models. Defaults to every model in the results.",
    )
    parser.add_argument("--good", type=int, default=2, help="Good examples per model.")
    parser.add_argument("--bad", type=int, default=2, help="Bad examples per model.")
    parser.add_argument(
        "--negatives",
        type=int,
        default=1,
        help="False-positive examples per model.",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text())
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    iou_threshold = float(payload["config"]["iou_threshold"])
    by_key = {
        (str(record["instance_id"]), Path(str(record["query"])).name): record
        for record in manifest
    }

    rows: list[dict[str, str]] = []
    for report in payload["models"]:
        model = str(report["model"])
        if args.models and model not in args.models:
            continue
        selected = select_examples(
            report["per_sample"],
            good_count=args.good,
            bad_count=args.bad,
            negative_count=args.negatives,
        )
        model_dir = args.output / SAFE_NAME.sub("_", model)
        model_dir.mkdir(parents=True, exist_ok=True)
        for index, (kind, sample) in enumerate(selected, start=1):
            key = (str(sample["instance_id"]), Path(str(sample["query"])).name)
            record = by_key.get(key)
            if record is None:
                continue
            image = render_example(
                root,
                record,
                sample,
                model=model,
                kind=kind,
                iou_threshold=iou_threshold,
            )
            name = (
                f"{kind}_{index:02d}_"
                f"{SAFE_NAME.sub('_', str(sample['instance_id']))}_"
                f"{Path(str(sample['query'])).stem}.png"
            )
            path = model_dir / name
            cv2.imwrite(str(path), image)
            rows.append(
                {
                    "model": model,
                    "kind": kind,
                    "class": str(sample["instance_id"]),
                    "query": Path(str(sample["query"])).name,
                    "iou": f"{float(sample['best_iou']):.3f}",
                    "hit": str(bool(sample["hit"])),
                    "matches": str(sample["match_count"]),
                    "inliers": str(sample["inlier_count"]),
                    "image": path.relative_to(args.output).as_posix(),
                }
            )

    index_path = args.output / "index.md"
    index_path.write_text(render_index(rows))
    print(f"wrote {len(rows)} examples to {args.output}")
    print(f"wrote {index_path}")
    return 0


def select_examples(
    records: Sequence[Mapping[str, object]],
    *,
    good_count: int,
    bad_count: int,
    negative_count: int,
) -> list[tuple[str, Mapping[str, object]]]:
    """Select good, bad, and negative examples.

    Args:
        records: Per-sample benchmark records.
        good_count: Number of successful examples.
        bad_count: Number of failed positive examples.
        negative_count: Number of false positives on negatives.

    Returns:
        ``(kind, record)`` pairs.
    """
    positives = [record for record in records if record["gt_boxes"]]
    negatives = [record for record in records if not record["gt_boxes"]]

    good = take_unique(
        sorted(positives, key=lambda record: float(record["best_iou"]), reverse=True),
        good_count,
    )
    used = {sample_key(record) for record in good}
    stages = [
        "no_matches",
        "too_few_matches",
        "ransac_failed",
        "degenerate_homography",
        "wrong_box",
    ]
    bad: list[Mapping[str, object]] = []
    for stage in stages:
        if len(bad) >= bad_count:
            break
        candidate = next(
            (
                record
                for record in positives
                if record.get("stage") == stage and sample_key(record) not in used
            ),
            None,
        )
        if candidate is not None:
            bad.append(candidate)
            used.add(sample_key(candidate))
    if len(bad) < bad_count:
        bad.extend(
            take_unique(
                [
                    record
                    for record in sorted(
                        positives,
                        key=lambda record: (
                            float(record["best_iou"]),
                            -int(record["inlier_count"]),
                        ),
                    )
                    if sample_key(record) not in used
                ],
                bad_count - len(bad),
            )
        )
        used.update(sample_key(record) for record in bad)
    false_positives = take_unique(
        [
            record
            for record in sorted(
                negatives,
                key=lambda record: int(record["inlier_count"]),
                reverse=True,
            )
            if record["predicted_box"] is not None and sample_key(record) not in used
        ],
        negative_count,
    )
    return (
        [("good", record) for record in good]
        + [(str(record.get("stage") or "bad"), record) for record in bad]
        + [("negative", record) for record in false_positives]
    )


def take_unique(
    records: Sequence[Mapping[str, object]],
    count: int,
) -> list[Mapping[str, object]]:
    """Take up to ``count`` records with distinct instances.

    Args:
        records: Candidate records.
        count: Maximum number to keep.

    Returns:
        Selected records.
    """
    selected: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for record in records:
        instance = str(record["instance_id"])
        if instance in seen:
            continue
        seen.add(instance)
        selected.append(record)
        if len(selected) == count:
            break
    return selected


def sample_key(record: Mapping[str, object]) -> tuple[str, str]:
    """Return a manifest lookup key for a per-sample record.

    Args:
        record: Per-sample record.

    Returns:
        ``(instance_id, query file name)``.
    """
    return (str(record["instance_id"]), Path(str(record["query"])).name)


def render_example(
    root: Path,
    record: Mapping[str, object],
    sample: Mapping[str, object],
    *,
    model: str,
    kind: str,
    iou_threshold: float,
) -> np.ndarray:
    """Render one reference/query example with boxes and metrics.

    Args:
        root: Dataset root.
        record: Manifest record.
        sample: Per-sample result.
        model: Model name.
        kind: Example kind.
        iou_threshold: Hit threshold shown in the footer.

    Returns:
        The rendered BGR image.
    """
    reference = load_image(root / str(record["reference"]))
    query = load_image(root / str(record["query"]))
    reference_box = record.get("reference_box")
    if reference_box is not None:
        reference = crop_box(reference, tuple(float(v) for v in reference_box))

    reference_display, _, _ = fit_image(reference, max_height=320, max_width=420)
    query_display, scale_x, scale_y = fit_image(query, max_height=520, max_width=760)

    draw_boxes(
        query_display,
        sample["gt_boxes"],
        scale_x,
        scale_y,
        color=(0, 180, 0),
        label="GT",
    )
    draw_boxes(
        query_display,
        [sample["predicted_box"]] if sample["predicted_box"] is not None else [],
        scale_x,
        scale_y,
        color=(0, 0, 230),
        label="pred",
    )

    header = 100
    footer = 42
    pad = 12
    reference_height, reference_width = reference_display.shape[:2]
    query_height, query_width = query_display.shape[:2]
    width = pad + reference_width + pad + query_width + pad
    height = header + max(reference_height, query_height) + footer
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)

    top = header
    canvas[top : top + reference_height, pad : pad + reference_width] = (
        reference_display
    )
    query_left = pad + reference_width + pad
    canvas[top : top + query_height, query_left : query_left + query_width] = (
        query_display
    )

    color = (30, 130, 30) if kind == "good" else (30, 30, 200)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), color, 3)
    text = [
        f"{model} | {kind} | {sample['instance_id']}",
        f"IoU={float(sample['best_iou']):.3f}  hit={bool(sample['hit'])}  "
        f"matches={sample['match_count']}  inliers={sample['inlier_count']}",
        f"query={Path(str(sample['query'])).name}  "
        f"latency={float(sample['latency_ms']):.1f} ms",
    ]
    for line_index, line in enumerate(text):
        cv2.putText(
            canvas,
            line,
            (pad, 26 + 24 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "reference crop",
        (pad, top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "query: GT=green, prediction=red",
        (query_left, top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    if sample["predicted_box"] is None:
        cv2.putText(
            canvas,
            "no prediction",
            (query_left + 8, top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 230),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"hit threshold: IoU >= {iou_threshold:.2f}",
        (pad, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_boxes(
    image: np.ndarray,
    boxes: Sequence[object],
    scale_x: float,
    scale_y: float,
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw scaled boxes on an image.

    Args:
        image: BGR image.
        boxes: Boxes in original coordinates.
        scale_x: Horizontal display scale.
        scale_y: Vertical display scale.
        color: BGR color.
        label: Box label.
    """
    height, width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box)
        left = max(0, min(width - 1, int(round(x1 * scale_x))))
        top = max(0, min(height - 1, int(round(y1 * scale_y))))
        right = max(0, min(width - 1, int(round(x2 * scale_x))))
        bottom = max(0, min(height - 1, int(round(y2 * scale_y))))
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(
            image,
            label,
            (left, max(12, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def fit_image(
    image: np.ndarray,
    *,
    max_height: int,
    max_width: int,
) -> tuple[np.ndarray, float, float]:
    """Resize an image to fit a display box.

    Args:
        image: BGR image.
        max_height: Maximum display height.
        max_width: Maximum display width.

    Returns:
        The resized image and its horizontal and vertical scales.
    """
    height, width = image.shape[:2]
    scale = min(1.0, max_height / height, max_width / width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, new_width / width, new_height / height


def crop_box(image: np.ndarray, box: Box) -> np.ndarray:
    """Crop an image to a box, clipped to image bounds.

    Args:
        image: BGR image.
        box: ``(x1, y1, x2, y2)`` box.

    Returns:
        The cropped image.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right <= left or bottom <= top:
        return image
    return image[top:bottom, left:right]


def load_image(path: Path) -> np.ndarray:
    """Load a BGR image.

    Args:
        path: Image path.

    Returns:
        The BGR image.

    Raises:
        FileNotFoundError: If the image cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def render_index(rows: Sequence[Mapping[str, str]]) -> str:
    """Render the examples index as Markdown.

    Args:
        rows: Example metadata rows.

    Returns:
        The Markdown document.
    """
    lines = [
        "# Prediction examples",
        "",
        "Good predictions, failures, and false positives from the benchmark.",
        "",
        "| Model | Kind | Class | Query | IoU | Hit | Matches | Inliers | Image |",
        "|---|---|---|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['kind']} | {row['class']} | {row['query']} "
            f"| {row['iou']} | {row['hit']} | {row['matches']} | {row['inliers']} "
            f"| [png]({row['image']}) |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
