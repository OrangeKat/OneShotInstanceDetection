#!/usr/bin/env python3
"""Evaluate the detector alone on real images: no matching involved.

The matcher is broken, so the benchmark measures it, not the detector. This
script isolates the detector with three real-image measurements:

- keypoint count per image;
- edge agreement: the percentile of image-gradient magnitude at each keypoint,
  against random pixels and against SIFT keypoints as a reference;
- object anchoring: the share of keypoints inside the annotated instance box.

It also writes a gallery so the numbers can be sanity-checked by eye.

Usage:
    uv run python scripts/bench/eval_detector.py --checkpoint outputs/train/heads.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from osid.model.train import load_checkpoint  # noqa: E402
from osid.utils.dataset import InstanceDataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the detector on real images."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gallery", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval/detector"),
    )
    return parser.parse_args(argv)


def gradient_rank(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return the gradient-magnitude percentile at each point.

    Args:
        gray: Grayscale image.
        points: ``(N, 2)`` pixel coordinates.

    Returns:
        ``(N,)`` percentile in ``[0, 1]``; 1 means the strongest edge.
    """
    magnitude = np.hypot(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    flat = np.sort(magnitude.ravel())
    rows = np.clip(points[:, 1].astype(int), 0, gray.shape[0] - 1)
    columns = np.clip(points[:, 0].astype(int), 0, gray.shape[1] - 1)
    values = magnitude[rows, columns]
    return np.searchsorted(flat, values) / max(len(flat) - 1, 1)


def main(argv: list[str] | None = None) -> int:
    """Run the detector evaluation.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    dataset = InstanceDataset(args.manifest)
    rng = np.random.default_rng(args.seed)
    positives = [s for s in dataset.samples if s.query_boxes]
    chosen = [
        positives[i] for i in rng.choice(len(positives), args.samples, replace=False)
    ]

    model = load_checkpoint(args.checkpoint, device=args.device)
    sift = cv2.SIFT_create(nfeatures=512)

    rows: list[dict[str, float]] = []
    gallery: list[np.ndarray] = []
    for index, sample in enumerate(dataset.load(s) for s in chosen):
        query = sample.query_image
        if query is None or not sample.query_boxes:
            continue
        gray = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)
        detections = model.detect(query)
        points = detections.keypoints
        box = max(sample.query_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        inside = (
            (
                (points[:, 0] >= box[0])
                & (points[:, 0] <= box[2])
                & (points[:, 1] >= box[1])
                & (points[:, 1] <= box[3])
            )
            if len(points)
            else np.zeros(0, dtype=bool)
        )
        ranks = gradient_rank(gray, points) if len(points) else np.zeros(0)

        keypoints, _ = sift.detectAndCompute(gray, None)
        sift_points = np.asarray([k.pt for k in keypoints[:512]], dtype=np.float32)
        sift_ranks = (
            gradient_rank(gray, sift_points) if len(sift_points) else np.zeros(0)
        )

        random_points = rng.uniform(
            [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1], (256, 2)
        )
        random_ranks = gradient_rank(gray, random_points)

        rows.append(
            {
                "instance_id": sample.instance_id,
                "query": sample.query_path.name,
                "keypoints": float(len(points)),
                "inside_box": float(inside.mean()) if len(inside) else 0.0,
                "gradient_rank": float(np.median(ranks)) if len(ranks) else 0.0,
                "sift_keypoints": float(len(sift_points)),
                "sift_gradient_rank": float(np.median(sift_ranks))
                if len(sift_ranks)
                else 0.0,
                "random_gradient_rank": float(np.median(random_ranks)),
            }
        )

        if len(gallery) < args.gallery:
            view = query.copy()
            for point in points:
                cv2.circle(view, (int(point[0]), int(point[1])), 2, (255, 140, 40), -1)
            cv2.rectangle(
                view,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (60, 220, 90),
                1,
            )
            gallery.append(view)

    def summarise(key: str) -> str:
        values = np.asarray([row[key] for row in rows])
        return f"median {np.median(values):.3f} | mean {values.mean():.3f}"

    print(f"images: {len(rows)}")
    print(f"keypoints per image:     {summarise('keypoints')}")
    print(f"inside GT box:           {summarise('inside_box')}")
    print(f"gradient percentile:     {summarise('gradient_rank')}")
    print(f"SIFT gradient percentile:{summarise('sift_gradient_rank')}")
    print(f"random gradient percent: {summarise('random_gradient_rank')}")
    print(f"SIFT keypoints:          {summarise('sift_keypoints')}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    if gallery:
        width = max(view.shape[1] for view in gallery)
        height = max(view.shape[0] for view in gallery)
        cells = []
        for view in gallery:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[: view.shape[0], : view.shape[1]] = view
            cells.append(canvas)
        columns = 3
        while len(cells) % columns:
            cells.append(np.zeros_like(cells[0]))
        grid = np.concatenate(
            [
                np.concatenate(cells[i : i + columns], axis=1)
                for i in range(0, len(cells), columns)
            ],
            axis=0,
        )
        cv2.imwrite(
            str(args.output / "gallery.png"),
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
        )
        print(f"wrote {args.output / 'gallery.png'}")
    print(f"wrote {args.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
