"""Detect keypoints on one image and write a visual check.

Usage:
    uv run python -m osid.model --index 0
    uv run python -m osid.model --image path/to/photo.jpg --top-k 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from osid.model.detector import Detections, DetectorDescriptor
from osid.utils.dataset import InstanceDataset

__all__ = ["main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detect keypoints on one image.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Image file to run on.")
    source.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
        help="Manifest used when --image is not given.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--instance")
    parser.add_argument("--query", help="Query file name for the manifest pair.")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Run on the reference crop instead of the query.",
    )
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
    """Run the detector on one image and write an overlay.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    image, label = _load_image(args)
    detector = DetectorDescriptor(
        device=args.device,
        max_keypoints=args.top_k,
    )
    detections = detector.detect(image)
    print(f"image:     {label}")
    print(f"keypoints: {len(detections)}")
    print(f"scores:    {_summary(detections.scores)}")
    print(
        f"descriptors: {detections.descriptors.shape}"
        f" | norm {np.linalg.norm(detections.descriptors, axis=1).mean():.3f}"
    )

    canvas = _overlay(image, detections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.output}")
    return 0


def _load_image(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """Load the requested image.

    Args:
        args: Parsed arguments.

    Returns:
        An RGB image and a label for the report.
    """
    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"could not read image: {args.image}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), str(args.image)

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
    if args.reference:
        return dataset.reference_crop(loaded), f"{sample.instance_id} (reference)"
    if loaded.query_image is None:
        raise SystemExit("query image failed to load")
    return loaded.query_image, f"{sample.instance_id} / {sample.query_path.name}"


def _summary(scores: np.ndarray) -> str:
    """Summarise boundary scores.

    Args:
        scores: Boundary probabilities.

    Returns:
        A short text summary.
    """
    if len(scores) == 0:
        return "none"
    return (
        f"n={len(scores)} min={scores.min():.3f} "
        f"median={np.median(scores):.3f} max={scores.max():.3f}"
    )


def _overlay(image: np.ndarray, detections: Detections) -> np.ndarray:
    """Draw keypoints, sized and coloured by boundary score.

    Args:
        image: RGB image.
        detections: Detector output.

    Returns:
        An RGB canvas.
    """
    canvas = image.copy()
    if len(detections) == 0:
        return canvas
    scores = detections.scores
    peak = max(float(scores.max()), 1e-6)
    for point, score in zip(detections.keypoints, scores):
        relative = score / peak
        colour = (
            int(255 * relative),
            int(180 * (1.0 - relative) + 60),
            int(80 * (1.0 - relative)),
        )
        cv2.circle(
            canvas,
            (int(point[0]), int(point[1])),
            int(2 + 3 * relative),
            colour,
            -1,
        )
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
