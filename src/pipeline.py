r"""Benchmark pipeline for one-shot instance detection.

Usage:
    uv run python -m src.pipeline --manifest data/flickrlogos32/manifest.json \\
        --models sift orb akaze --output outputs/flickrlogos32
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.models.feature_matching import AKAZEMatcher, ORBMatcher, SIFTMatcher
from src.models.feature_matching.base import FeatureMatcher, MatchResult
from src.utils.dataset import Box, InstanceDataset, InstanceSample
from src.utils.logger import get_logger

__all__ = ["main"]

MODELS: dict[str, type[FeatureMatcher]] = {
    "sift": SIFTMatcher,
    "orb": ORBMatcher,
    "akaze": AKAZEMatcher,
}


@dataclass
class SampleResult:
    """Per-sample benchmark record.

    Attributes:
        dataset: Dataset name.
        instance_id: Instance identifier.
        query: Query image path.
        latency_ms: Matching latency in milliseconds.
        match_count: Matches after the ratio test.
        inlier_count: RANSAC inliers.
        best_iou: Best IoU against the ground-truth boxes.
        predicted_box: Predicted query box, or ``None``.
        gt_boxes: Ground-truth query boxes.
        hit: Whether the best IoU reaches the hit threshold.
    """

    dataset: str
    instance_id: str
    query: str
    latency_ms: float
    match_count: int
    inlier_count: int
    best_iou: float
    predicted_box: Box | None
    gt_boxes: tuple[Box, ...]
    hit: bool


def box_iou(first: Box, second: Box) -> float:
    """Compute the intersection-over-union of two boxes.

    Args:
        first: First ``(x1, y1, x2, y2)`` box.
        second: Second ``(x1, y1, x2, y2)`` box.

    Returns:
        The IoU in ``[0, 1]``.
    """
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def predicted_box(match: MatchResult, reference_shape: tuple[int, int]) -> Box | None:
    """Map the reference crop corners through the estimated homography.

    Args:
        match: Match result with an optional homography.
        reference_shape: ``(height, width)`` of the reference crop.

    Returns:
        The axis-aligned predicted query box, or ``None``.
    """
    if match.homography is None:
        return None
    height, width = reference_shape
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), match.homography
    ).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    if x2 <= x1 or y2 <= y1:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def _reference_crop(sample: InstanceSample) -> np.ndarray:
    """Return the RGB reference crop for a sample.

    Args:
        sample: Reference/query pair.

    Returns:
        The cropped reference image.
    """
    image = sample.reference_image
    if image is None:
        image = _load_rgb(sample.reference_path)
    if sample.reference_box is None:
        return image
    height, width = image.shape[:2]
    x1, y1, x2, y2 = sample.reference_box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return image[top:bottom, left:right]


def _query_image(sample: InstanceSample) -> np.ndarray:
    """Return the RGB query image for a sample.

    Args:
        sample: Reference/query pair.

    Returns:
        The query image.
    """
    if sample.query_image is not None:
        return sample.query_image
    return _load_rgb(sample.query_path)


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB.

    Args:
        path: Image path.

    Returns:
        The RGB image.

    Raises:
        FileNotFoundError: If the image cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def average_precision(records: Sequence[SampleResult], threshold: float) -> float:
    """Compute 11-point average precision at an IoU threshold.

    Args:
        records: Per-sample records.
        threshold: IoU threshold that defines a true positive.

    Returns:
        The average precision.
    """
    positives = sum(1 for record in records if record.gt_boxes)
    if positives == 0:
        return 0.0
    detections = [record for record in records if record.predicted_box is not None]
    detections.sort(key=lambda r: (r.inlier_count, r.match_count), reverse=True)
    true_positives = 0
    false_positives = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for record in detections:
        if record.gt_boxes and record.best_iou >= threshold:
            true_positives += 1
        else:
            false_positives += 1
        precisions.append(true_positives / (true_positives + false_positives))
        recalls.append(true_positives / positives)
    total = 0.0
    for point in np.linspace(0.0, 1.0, 11):
        best = max(
            (
                precision
                for precision, recall in zip(precisions, recalls)
                if recall >= point
            ),
            default=0.0,
        )
        total += best
    return total / 11.0


def summarize(
    name: str,
    records: Sequence[SampleResult],
    iou_threshold: float,
) -> dict[str, object]:
    """Aggregate per-sample records into a model report.

    Args:
        name: Model name.
        records: Per-sample records.
        iou_threshold: IoU threshold that defines a hit.

    Returns:
        A JSON-ready report.
    """
    positives = [record for record in records if record.gt_boxes]
    hits = sum(1 for record in positives if record.hit)
    latencies = np.asarray([record.latency_ms for record in records], dtype=np.float64)
    mean_iou = (
        float(np.mean([record.best_iou for record in positives])) if positives else 0.0
    )
    false_positives = sum(
        1 for record in records if record.predicted_box is not None and not record.hit
    )
    return {
        "model": name,
        "samples": len(records),
        "positives": len(positives),
        "hit_rate": hits / len(positives) if positives else 0.0,
        "ap50": average_precision(records, iou_threshold),
        "mean_iou": mean_iou,
        "mean_matches": (
            float(np.mean([record.match_count for record in records]))
            if records
            else 0.0
        ),
        "mean_inliers": (
            float(np.mean([record.inlier_count for record in records]))
            if records
            else 0.0
        ),
        "false_positives": false_positives,
        "latency_ms": {
            "mean": float(latencies.mean()) if len(latencies) else 0.0,
            "p50": float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
            "p95": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        },
        "per_sample": [_record_dict(record) for record in records],
    }


def run_benchmark(
    matcher: FeatureMatcher,
    samples: Sequence[InstanceSample],
    iou_threshold: float,
) -> dict[str, object]:
    """Run one matcher over all samples.

    Args:
        matcher: Model instance.
        samples: Reference/query pairs.
        iou_threshold: IoU threshold that defines a hit.

    Returns:
        A model report.
    """
    records: list[SampleResult] = []
    for sample in samples:
        reference = _reference_crop(sample)
        query = _query_image(sample)
        start = time.perf_counter()
        match = matcher.match(reference, query)
        latency_ms = (time.perf_counter() - start) * 1000.0
        box = predicted_box(match, reference.shape[:2])
        best_iou = (
            max(box_iou(box, gt) for gt in sample.query_boxes)
            if box is not None and sample.query_boxes
            else 0.0
        )
        records.append(
            SampleResult(
                dataset=sample.dataset,
                instance_id=sample.instance_id,
                query=str(sample.query_path),
                latency_ms=latency_ms,
                match_count=match.match_count,
                inlier_count=match.inlier_count,
                best_iou=best_iou,
                predicted_box=box,
                gt_boxes=sample.query_boxes,
                hit=best_iou >= iou_threshold,
            )
        )
    return summarize(matcher.name, records, iou_threshold)


def _record_dict(record: SampleResult) -> dict[str, object]:
    """Convert a per-sample record to JSON-ready data.

    Args:
        record: Per-sample record.

    Returns:
        A JSON-ready dictionary.
    """
    return {
        "dataset": record.dataset,
        "instance_id": record.instance_id,
        "query": record.query,
        "latency_ms": record.latency_ms,
        "match_count": record.match_count,
        "inlier_count": record.inlier_count,
        "best_iou": record.best_iou,
        "predicted_box": _box_list(record.predicted_box),
        "gt_boxes": [_box_list(box) for box in record.gt_boxes],
        "hit": record.hit,
    }


def _box_list(box: Box | None) -> list[float] | None:
    """Convert an optional box to a JSON-ready list.

    Args:
        box: Optional box.

    Returns:
        A list of four floats, or ``None``.
    """
    return None if box is None else [float(value) for value in box]


def write_reports(
    reports: Sequence[dict[str, object]],
    config: dict[str, object],
    output: Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports.

    Args:
        reports: Model reports.
        config: Run configuration.
        output: Output directory.

    Returns:
        The JSON and Markdown paths.
    """
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    json_path.write_text(json.dumps({"config": config, "models": reports}, indent=2))
    markdown_path = output / "results.md"
    markdown_path.write_text(_markdown(reports, config))
    return json_path, markdown_path


def _markdown(
    reports: Sequence[dict[str, object]],
    config: dict[str, object],
) -> str:
    """Render a Markdown summary.

    Args:
        reports: Model reports.
        config: Run configuration.

    Returns:
        The Markdown document.
    """
    lines = [
        "# Benchmark results",
        "",
        f"- Manifest: `{config['manifest']}`",
        f"- Samples: `{config['samples']}`",
        f"- IoU threshold: `{config['iou_threshold']}`",
        f"- Max keypoints: `{config['max_keypoints']}`",
        f"- Ratio: `{config['ratio']}`",
        f"- RANSAC threshold: `{config['ransac_threshold']}`",
        "",
        "| Model | Samples | Hit rate | AP@0.5 | Mean IoU | Matches | Inliers "
        "| p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        latency = report["latency_ms"]
        lines.append(
            f"| {report['model']} | {report['samples']} | {report['hit_rate']:.3f} "
            f"| {report['ap50']:.3f} | {report['mean_iou']:.3f} "
            f"| {report['mean_matches']:.1f} | {report['mean_inliers']:.1f} "
            f"| {latency['p50']:.1f} | {latency['p95']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list, defaulting to ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="One-shot instance detection benchmark."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON or JSONL manifest.",
    )
    parser.add_argument("--root", type=Path, help="Base directory for manifest paths.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODELS, "all"],
        default=["all"],
        help="Models to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark"),
        help="Output directory for results.json and results.md.",
    )
    parser.add_argument("--max-samples", type=int, help="Limit the number of samples.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for a hit.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold in pixels.",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Keypoint budget per image.",
    )
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio.")
    return parser.parse_args(argv)


def _selected_models(names: Sequence[str]) -> list[str]:
    """Return the model names to run, preserving order.

    Args:
        names: Requested model names, possibly including ``"all"``.

    Returns:
        De-duplicated model names.
    """
    if "all" in names:
        return list(MODELS)
    selected: list[str] = []
    for name in names:
        if name not in selected:
            selected.append(name)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark pipeline.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    log = get_logger(__name__)

    dataset = InstanceDataset(args.manifest, root=args.root)
    samples = list(dataset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    names = _selected_models(args.models)
    config: dict[str, object] = {
        "manifest": str(args.manifest),
        "models": names,
        "samples": len(samples),
        "iou_threshold": args.iou_threshold,
        "max_keypoints": args.max_keypoints,
        "ratio": args.ratio,
        "ransac_threshold": args.ransac_threshold,
    }
    reports: list[dict[str, object]] = []
    for name in names:
        matcher = MODELS[name](
            max_keypoints=args.max_keypoints,
            ratio=args.ratio,
            ransac_threshold=args.ransac_threshold,
        )
        report = run_benchmark(matcher, samples, args.iou_threshold)
        reports.append(report)
        latency = report["latency_ms"]
        log.info(
            "%s: hit=%.3f ap50=%.3f iou=%.3f p50=%.1fms",
            name,
            report["hit_rate"],
            report["ap50"],
            report["mean_iou"],
            latency["p50"],
        )

    json_path, markdown_path = write_reports(reports, config, args.output)
    log.info("wrote %s", json_path)
    log.info("wrote %s", markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
