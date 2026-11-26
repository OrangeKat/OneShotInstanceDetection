#!/usr/bin/env python3
"""Phase 0, oracle test: does restricting matches to the object fix the geometry?

Filters a matcher's correspondences down to the ones whose query point lands
inside the GT box, re-estimates the homography from those, and compares the
resulting box against the original. The GT box is an oracle — this is an upper
bound on what object-anchored keypoints could buy, not a runnable method.

Usage:
    uv run python scripts/analysis/oracle_object_matches.py --sample 40
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from osid.benchmark.matchers.base import estimate_homography  # noqa: E402
from osid.benchmark.registry import MODELS  # noqa: E402
from osid.utils.dataset import InstanceDataset  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Oracle object-match filtering.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
    )
    parser.add_argument(
        "--hard-pairs",
        type=Path,
        default=Path("outputs/flickrlogos32-refs/analysis/hard_pairs.csv"),
    )
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--models", nargs="+", default=["sift", "superpoint_lightglue"])
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/week4/oracle_object_matches.csv"),
    )
    return parser.parse_args(argv)


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the IoU of two boxes.

    Args:
        left: First box.
        right: Second box.

    Returns:
        IoU in ``[0, 1]``.
    """
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def project(homography: np.ndarray, shape: tuple[int, int]) -> tuple[float, ...] | None:
    """Project the reference crop corners through a homography.

    Args:
        homography: ``3 x 3`` transform.
        shape: Reference ``(height, width)``.

    Returns:
        Axis-aligned box, or ``None``.
    """
    height, width = shape
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners.reshape(1, -1, 2), homography).reshape(
        -1, 2
    )
    if not np.isfinite(projected).all():
        return None
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    return (float(x1), float(y1), float(x2), float(y2))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the oracle test.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    dataset = InstanceDataset(args.manifest)
    by_key = {
        (s.instance_id, s.query_path.name): index
        for index, s in enumerate(dataset.samples)
    }
    hard = list(csv.DictReader(args.hard_pairs.open()))
    rng = np.random.default_rng(0)
    sample = [
        hard[i]
        for i in rng.choice(len(hard), min(args.sample, len(hard)), replace=False)
    ]

    rows: list[dict[str, object]] = []
    for row in sample:
        key = (row["class_name"], row["query"])
        index = by_key.get(key)
        if index is None:
            continue
        sample_row = dataset[index]
        reference = dataset.reference_crop(sample_row)
        image = sample_row.query_image
        if image is None or not sample_row.query_boxes:
            continue
        gt = max(
            sample_row.query_boxes,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        )
        for model_name in args.models:
            matcher = MODELS[model_name](max_keypoints=args.max_keypoints)
            result = matcher.match(reference, image)
            if not result.matches:
                continue
            points0 = result.keypoints0[[index0 for index0, _ in result.matches]]
            points1 = result.keypoints1[[index1 for _, index1 in result.matches]]
            inside = (
                (points1[:, 0] >= gt[0])
                & (points1[:, 0] <= gt[2])
                & (points1[:, 1] >= gt[1])
                & (points1[:, 1] <= gt[3])
            )
            # The claim is about outline points, which span the object, so the
            # test compares three selections: the box interior (clustered), a
            # widened box, and a band around the box edge (spanning).
            cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
            half_w, half_h = (gt[2] - gt[0]) / 2, (gt[3] - gt[1]) / 2

            def scaled(factor: float) -> np.ndarray:
                return np.array(
                    [
                        cx - half_w * factor,
                        cy - half_h * factor,
                        cx + half_w * factor,
                        cy + half_h * factor,
                    ]
                )

            widened_box = scaled(1.35)
            inner_box = scaled(0.75)
            spanned = (
                (points1[:, 0] >= widened_box[0])
                & (points1[:, 0] <= widened_box[2])
                & (points1[:, 1] >= widened_box[1])
                & (points1[:, 1] <= widened_box[3])
            )
            ring = spanned & ~(
                (points1[:, 0] >= inner_box[0])
                & (points1[:, 0] <= inner_box[2])
                & (points1[:, 1] >= inner_box[1])
                & (points1[:, 1] <= inner_box[3])
            )

            def iou_for(selection: np.ndarray) -> float:
                if int(selection.sum()) < 4:
                    return 0.0
                matrix, _, _ = estimate_homography(
                    points0[selection],
                    points1[selection],
                    args.ransac_threshold,
                )
                if matrix is None:
                    return 0.0
                projected = project(matrix, reference.shape[:2])
                return 0.0 if projected is None else box_iou(projected, gt)

            original = iou_for(np.ones(len(points0), dtype=bool))
            filtered = iou_for(inside)
            widened = iou_for(spanned)
            band = iou_for(ring)
            rows.append(
                {
                    "instance_id": row["class_name"],
                    "query": row["query"],
                    "model": model_name,
                    "matches": len(points0),
                    "inside_share": round(float(inside.mean()), 3),
                    "iou_all": round(original, 3),
                    "iou_inside": round(filtered, 3),
                    "iou_widened": round(widened, 3),
                    "iou_ring": round(band, 3),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'model':22s} {'pairs':>5s} {'inside':>7s} {'IoU all':>8s} {'IoU obj':>8s} {'hit all':>8s} {'hit obj':>8s}"
    )
    for model_name in args.models:
        subset = [row for row in rows if row["model"] == model_name]
        if not subset:
            continue
        all_iou = np.array([row["iou_all"] for row in subset])
        obj_iou = np.array([row["iou_inside"] for row in subset])
        inside = np.mean([row["inside_share"] for row in subset])
        print(
            f"{model_name:22s} {len(subset):5d} {inside:7.3f} "
            f"{np.median(all_iou):8.3f} {np.median(obj_iou):8.3f} "
            f"{np.mean(all_iou >= 0.5):8.3f} {np.mean(obj_iou >= 0.5):8.3f}"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
