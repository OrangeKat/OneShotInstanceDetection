#!/usr/bin/env python3
"""Phase 0: where do keypoints and matches land relative to the instance?

The model design claims that matching fails because keypoints and consensus
matches sit on background rather than on the object. This measures that claim
directly: run a matcher on a sample of pairs, then split its query keypoints and
its accepted matches into inside versus outside the GT box.

A pair where RANSAC keeps mostly outside-the-box matches is evidence for the
background hypothesis; a pair where the inliers are inside but the box is
misplaced is evidence for a geometry problem instead.

Usage:
    uv run python scripts/analysis/keypoint_placement.py --sample 40
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

import numpy as np  # noqa: E402

from osid.benchmark.registry import MODELS  # noqa: E402
from osid.utils.dataset import InstanceDataset  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Measure keypoint placement.")
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
    parser.add_argument(
        "--agreement",
        type=Path,
        default=Path("outputs/flickrlogos32-refs/analysis/pair_agreement.csv"),
    )
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--models", nargs="+", default=["sift", "superpoint_lightglue"])
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/week4/keypoint_placement.csv"),
    )
    return parser.parse_args(argv)


def inside(point: np.ndarray, box: Sequence[float]) -> bool:
    """Report whether a point lies inside a box.

    Args:
        point: ``(x, y)`` coordinate.
        box: ``(x1, y1, x2, y2)`` box.

    Returns:
        True when the point is inside.
    """
    return bool(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement.

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

    hard = [
        (row["class_name"], row["query"])
        for row in csv.DictReader(args.hard_pairs.open())
    ]
    easy = [
        (row["class_name"], row["query"])
        for row in csv.DictReader(args.agreement.open())
        if row["negative"] == "False" and int(row["models_hit"]) == 8
    ]
    rng = np.random.default_rng(0)
    hard_sample = [
        hard[i]
        for i in rng.choice(len(hard), min(args.sample, len(hard)), replace=False)
    ]
    easy_sample = [
        easy[i]
        for i in rng.choice(len(easy), min(args.sample, len(easy)), replace=False)
    ]

    records: list[dict[str, object]] = []
    for label, sample in (("hard", hard_sample), ("easy", easy_sample)):
        for class_name, query in sample:
            index = by_key.get((class_name, query))
            if index is None:
                continue
            sample_row = dataset[index]
            reference = dataset.reference_crop(sample_row)
            image = sample_row.query_image
            if image is None or not sample_row.query_boxes:
                continue
            box = max(
                sample_row.query_boxes,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            )
            for model_name in args.models:
                matcher = MODELS[model_name](max_keypoints=args.max_keypoints)
                result = matcher.match(reference, image)
                if not result.matches:
                    continue
                mask = (
                    list(result.inlier_mask)
                    if result.inlier_mask is not None
                    else [False] * len(result.matches)
                )
                points = [result.keypoints1[index1] for _, index1 in result.matches]
                flags = [inside(point, box) for point in points]
                inliers = [flag for flag, keep in zip(flags, mask) if keep]
                outliers = [flag for flag, keep in zip(flags, mask) if not keep]
                keypoints_inside = sum(
                    inside(point, box) for point in result.keypoints1
                )
                records.append(
                    {
                        "group": label,
                        "instance_id": class_name,
                        "query": query,
                        "model": model_name,
                        "matches": len(flags),
                        "inliers": len(inliers),
                        "matches_inside": round(float(np.mean(flags)), 3),
                        "inliers_inside": round(float(np.mean(inliers)), 3)
                        if inliers
                        else "",
                        "outliers_inside": round(float(np.mean(outliers)), 3)
                        if outliers
                        else "",
                        "keypoints_inside": round(
                            keypoints_inside / max(1, len(result.keypoints1)), 3
                        ),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(
        f"{'group':6s} {'model':22s} {'pairs':>5s} {'kp inside':>10s} {'match inside':>13s} {'inlier inside':>14s}"
    )
    for label in ("hard", "easy"):
        for model_name in args.models:
            subset = [
                r for r in records if r["group"] == label and r["model"] == model_name
            ]
            if not subset:
                continue
            kp = np.mean([float(r["keypoints_inside"]) for r in subset])
            mt = np.mean([float(r["matches_inside"]) for r in subset])
            inl = [
                float(r["inliers_inside"]) for r in subset if r["inliers_inside"] != ""
            ]
            print(
                f"{label:6s} {model_name:22s} {len(subset):5d} {kp:10.3f} {mt:13.3f} "
                f"{np.mean(inl) if inl else float('nan'):14.3f}"
            )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
