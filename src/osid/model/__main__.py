"""Run the model on one benchmark pair and save a visual check.

Usage:
    uv run python -m osid.model --index 0 --model sift
    uv run python -m osid.model --instance carlsberg --query 145074777.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from osid.model.model import BoundaryModel
from osid.utils.dataset import InstanceDataset

__all__ = ["main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualise the model on one pair.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--instance")
    parser.add_argument("--query")
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model-demo.png"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the model on one pair and write an overlay.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    dataset = InstanceDataset(args.manifest)
    sample = None
    if args.instance and args.query:
        for candidate in dataset.samples:
            if (
                candidate.instance_id == args.instance
                and candidate.query_path.name == args.query
            ):
                sample = candidate
                break
    else:
        sample = dataset.samples[args.index]
    if sample is None:
        raise SystemExit("pair not found in the manifest")

    loaded = dataset.load(sample)
    reference = dataset.reference_crop(loaded)
    query = loaded.query_image
    if query is None:
        raise SystemExit("query image failed to load")

    model = BoundaryModel(
        device=args.device,
        max_keypoints=args.top_k,
    )
    result = model.match(reference, query)
    print(f"pair: {sample.instance_id} / {sample.query_path.name}")
    print(f"reference keypoints: {len(result.keypoints0)}")
    print(f"query keypoints:     {len(result.keypoints1)}")
    print(f"matches:             {result.match_count} (inliers {result.inlier_count})")
    print(f"reference scores:    {_summary(result.keypoint_scores0)}")
    print(f"query scores:        {_summary(result.keypoint_scores1)}")

    canvas = _overlay(reference, query, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.output}")
    return 0


def _summary(scores: tuple[float, ...]) -> str:
    """Summarise a score tuple.

    Args:
        scores: Boundary probabilities.

    Returns:
        A short text summary.
    """
    if not scores:
        return "none"
    values = np.asarray(scores)
    return (
        f"n={len(values)} min={values.min():.3f} "
        f"median={np.median(values):.3f} max={values.max():.3f}"
    )


def _overlay(reference: np.ndarray, query: np.ndarray, result: object) -> np.ndarray:
    """Draw keypoints on both images side by side.

    Args:
        reference: Reference crop.
        query: Query image.
        result: Match result.

    Returns:
        An RGB canvas.
    """
    keypoints0 = getattr(result, "keypoints0")
    keypoints1 = getattr(result, "keypoints1")
    scores0 = getattr(result, "keypoint_scores0")
    scores1 = getattr(result, "keypoint_scores1")

    reference_view = reference.copy()
    for point, score in zip(keypoints0, scores0):
        cv2.circle(
            reference_view,
            (int(point[0]), int(point[1])),
            2,
            (60, 180, 255),
            -1,
        )
        if score > 0.5:
            cv2.circle(
                reference_view,
                (int(point[0]), int(point[1])),
                5,
                (255, 220, 60),
                1,
            )
    query_view = query.copy()
    for point, score in zip(keypoints1, scores1):
        cv2.circle(query_view, (int(point[0]), int(point[1])), 2, (80, 255, 160), -1)

    height = max(reference_view.shape[0], query_view.shape[0])
    width = reference_view.shape[1] + query_view.shape[1] + 8
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    canvas[: reference_view.shape[0], : reference_view.shape[1]] = reference_view
    canvas[
        : query_view.shape[0],
        reference_view.shape[1] + 8 :,
    ] = query_view
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
