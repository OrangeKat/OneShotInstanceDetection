#!/usr/bin/env python3
"""Diagnose a single reference/query pair.

Runs one matcher on one manifest pair and writes a match overlay plus a JSON
record with stage, geometry, stability, and optional oracle results.

Usage:
    uv run python scripts/diagnose.py --model sift --index 0
    uv run python scripts/diagnose.py --model roma --instance adidas-1 --query 0001.jpg
    uv run python scripts/diagnose.py --model sift --index 0 --seeds 5 --oracle
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.feature_matching.base import FeatureMatcher
from src.pipeline import MODELS, box_iou, predicted_box
from src.utils.dataset import InstanceDataset, InstanceSample
from src.utils.diagnostics import Box, match_diagnostics
from src.utils.overlay import render_overlay
from src.utils.seeding import set_seed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the single-pair diagnostic.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Diagnose one benchmark pair.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
        help="Benchmark manifest.",
    )
    parser.add_argument("--model", choices=sorted(MODELS), default="sift")
    parser.add_argument("--index", type=int, default=0, help="Manifest index.")
    parser.add_argument("--instance", help="Instance id (alternative to --index).")
    parser.add_argument("--query", help="Query file name (alternative to --index).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/diagnose"),
        help="Directory for the overlay and JSON.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Repeat RANSAC with this many seeds.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Run reference-scale and query-GT-crop oracles.",
    )
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    args = parser.parse_args(argv)

    dataset = InstanceDataset(args.manifest, load_images=False)
    sample = dataset.load(select_sample(dataset.samples, args))
    reference = dataset.reference_crop(sample)
    query = sample.query_image
    if query is None:
        raise SystemExit("query image failed to load")

    matcher = MODELS[args.model](
        max_keypoints=args.max_keypoints,
        ransac_threshold=args.ransac_threshold,
    )
    set_seed(0)
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
    record: dict[str, object] = {
        "model": args.model,
        "instance_id": sample.instance_id,
        "query": str(sample.query_path),
        "reference": str(sample.reference_path),
        "reference_box": sample.reference_box,
        "query_boxes": sample.query_boxes,
        "predicted_box": box,
        "best_iou": iou,
        **diagnostics,
    }
    if args.seeds > 1:
        record["stability"] = stability(
            matcher,
            reference,
            query,
            sample.query_boxes,
            args.seeds,
        )
    if args.oracle:
        record["oracle_scale"] = scale_oracle(
            matcher,
            reference,
            query,
            sample.query_boxes,
        )
        record["oracle_query_crop"] = query_crop_oracle(matcher, sample)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{sample.instance_id}_{sample.query_path.stem}"
    overlay = render_overlay(
        reference,
        query,
        result,
        box,
        sample.query_boxes,
        diagnostics,
        iou,
    )
    cv2.imwrite(str(args.output / f"{stem}_overlay.png"), overlay)
    (args.output / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"wrote {args.output / f'{stem}_overlay.png'}")
    return 0


def select_sample(
    samples: Sequence[InstanceSample],
    args: argparse.Namespace,
) -> InstanceSample:
    """Select a sample by index, instance, or query.

    Args:
        samples: Dataset samples.
        args: Parsed arguments.

    Returns:
        Selected sample.

    Raises:
        SystemExit: If no sample matches.
    """
    if args.instance is not None:
        for sample in samples:
            if sample.instance_id == args.instance and (
                args.query is None or sample.query_path.name == args.query
            ):
                return sample
    if args.index < len(samples) and args.instance is None:
        return samples[args.index]
    raise SystemExit("no sample matched --instance/--query/--index")


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


def stability(
    matcher: FeatureMatcher,
    reference: np.ndarray,
    query: np.ndarray,
    gt_boxes: Sequence[Box],
    seeds: int,
) -> dict[str, object]:
    """Repeat matching with several RANSAC seeds.

    Args:
        matcher: Feature matcher.
        reference: Reference crop.
        query: Query image.
        gt_boxes: Ground-truth boxes.
        seeds: Number of seeds.

    Returns:
        Stability summary.
    """
    ious: list[float] = []
    centers: list[tuple[float, float]] = []
    for seed in range(seeds):
        set_seed(seed)
        result = matcher.match(reference, query)
        box = predicted_box(result, reference.shape[:2])
        ious.append(best_iou(box, gt_boxes))
        if box is not None:
            centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    center_array = np.asarray(centers, dtype=np.float64)
    return {
        "seeds": seeds,
        "hit_rate": float(np.mean([iou >= 0.5 for iou in ious])),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "box_center_std": (
            float(np.linalg.norm(center_array.std(axis=0)))
            if len(center_array) > 1
            else 0.0
        ),
        "ious": ious,
    }


def scale_oracle(
    matcher: FeatureMatcher,
    reference: np.ndarray,
    query: np.ndarray,
    gt_boxes: Sequence[Box],
) -> dict[str, object]:
    """Resize the reference to the GT box scale and rematch.

    Args:
        matcher: Feature matcher.
        reference: Reference crop.
        query: Query image.
        gt_boxes: Ground-truth boxes.

    Returns:
        Oracle result.
    """
    if not gt_boxes:
        return {"available": False}
    gt = max(
        gt_boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
    )
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    ref_area = reference.shape[0] * reference.shape[1]
    factor = float(np.clip(np.sqrt(gt_area / max(1.0, ref_area)), 0.1, 10.0))
    resized = cv2.resize(
        reference,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_AREA,
    )
    result = matcher.match(resized, query)
    box = predicted_box(result, resized.shape[:2])
    iou = best_iou(box, gt_boxes)
    return {
        "available": True,
        "scale_factor": factor,
        "best_iou": iou,
        "hit": iou >= 0.5,
    }


def query_crop_oracle(
    matcher: FeatureMatcher,
    sample: InstanceSample,
) -> dict[str, object]:
    """Match a GT crop from the query to the reference image.

    Args:
        matcher: Feature matcher.
        sample: Reference/query pair.
        reference: Reference crop.

    Returns:
        Oracle result.
    """
    if not sample.query_boxes or sample.query_image is None:
        return {"available": False}
    gt = max(
        sample.query_boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
    )
    x1, y1, x2, y2 = (int(round(value)) for value in gt)
    crop = sample.query_image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"available": False}
    reference_image = cv2.imread(str(sample.reference_path))
    if reference_image is None:
        return {"available": False}
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    result = matcher.match(crop, reference_image)
    box = predicted_box(result, crop.shape[:2])
    target = sample.reference_box
    if target is None:
        height, width = reference_image.shape[:2]
        target = (0.0, 0.0, float(width), float(height))
    iou = best_iou(box, (target,))
    return {"available": True, "best_iou": iou, "hit": iou >= 0.5}


if __name__ == "__main__":
    raise SystemExit(main())
