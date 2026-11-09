r"""Benchmark pipeline for one-shot instance detection and matcher validation.

Usage:
    uv run python -m src.pipeline --manifest data/flickrlogos32/manifest.json \\
        --models sift orb akaze --output outputs/flickrlogos32
    uv run python -m src.pipeline --protocol hpatches \\
        --models sift orb akaze --output outputs/hpatches
    uv run python -m src.pipeline --protocol retrieval \\
        --models sift orb akaze --output outputs/paris6k
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.feature_matching import (
    AKAZEMatcher,
    ALIKEDLightGlueMatcher,
    FlexGlueMatcher,
    LoFTRMatcher,
    ORBMatcher,
    RoMaMatcher,
    SIFTMatcher,
    SuperPointLightGlueMatcher,
)
from src.models.feature_matching.base import FeatureMatcher, MatchResult
from src.utils.dataset import Box, InstanceDataset, InstanceSample
from src.utils.logger import get_logger
from src.validation.hpatches import evaluate_model as evaluate_hpatches
from src.validation.hpatches import load_pairs
from src.validation.report import render_report
from src.validation.retrieval import evaluate_model as evaluate_retrieval
from src.validation.retrieval import load_queries
from src.validation.retrieval_report import render_retrieval_report

__all__ = ["main"]

MODELS: dict[str, type[FeatureMatcher]] = {
    "sift": SIFTMatcher,
    "orb": ORBMatcher,
    "akaze": AKAZEMatcher,
    "superpoint_lightglue": SuperPointLightGlueMatcher,
    "aliked_lightglue": ALIKEDLightGlueMatcher,
    "loftr": LoFTRMatcher,
    "roma": RoMaMatcher,
    "flexglue": FlexGlueMatcher,
}


def _configure_gpu(fraction: float, log: Logger) -> None:
    """Cap this process's GPU memory so a runaway model cannot hang the driver.

    Args:
        fraction: Requested share of total VRAM, or ``0`` to auto-size from
            the currently free memory.
        log: Logger.
    """
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if fraction > 0:
        budget = min(fraction * total, 0.9 * free)
    else:
        budget = min(0.9 * total, 0.9 * free)
    budget = min(max(budget, 512 * 1024**2), total)
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    log.info(
        "GPU memory budget: %.1f GiB (%.1f GiB free of %.1f GiB)",
        budget / 1024**3,
        free / 1024**3,
        total / 1024**3,
    )
    if free < 2 * 1024**3:
        log.warning(
            "less than 2 GiB of GPU memory is free; close other GPU apps or "
            "lower --max-side"
        )


def _release_gpu() -> None:
    """Drop model references and return cached blocks to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_gpu_peak() -> None:
    """Reset the GPU peak-memory counters used in reports."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_memory_stats() -> dict[str, float]:
    """Return peak GPU memory for the current model, in MiB."""
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_gpu_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 1024**2,
            1,
        ),
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
    loader: Callable[[InstanceSample], InstanceSample] | None = None,
) -> dict[str, object]:
    """Run one matcher over all samples.

    Args:
        matcher: Model instance.
        samples: Reference/query pairs.
        iou_threshold: IoU threshold that defines a hit.
        loader: Optional callable that loads a sample's images.

    Returns:
        A model report.
    """
    records: list[SampleResult] = []
    for sample in samples:
        if loader is not None:
            sample = loader(sample)
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


def write_hpatches_reports(
    reports: Sequence[dict[str, object]],
    config: dict[str, object],
    output: Path,
) -> tuple[Path, Path]:
    """Write HPatches JSON and Markdown reports.

    Args:
        reports: Per-model HPatches metrics.
        config: Run configuration.
        output: Output directory.

    Returns:
        The JSON and Markdown paths.
    """
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    json_path.write_text(
        json.dumps({"config": config, "models": reports}, indent=2) + "\n"
    )
    markdown_path = output / "results.md"
    markdown_path.write_text(render_report(reports, config))
    return json_path, markdown_path


def write_retrieval_reports(
    reports: Sequence[dict[str, object]],
    config: dict[str, object],
    output: Path,
) -> tuple[Path, Path]:
    """Write retrieval JSON and Markdown reports.

    Args:
        reports: Per-model retrieval metrics.
        config: Run configuration.
        output: Output directory.

    Returns:
        The JSON and Markdown paths.
    """
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    json_path.write_text(
        json.dumps({"config": config, "models": reports}, indent=2) + "\n"
    )
    markdown_path = output / "results.md"
    markdown_path.write_text(render_retrieval_report(reports, config))
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
        f"- Max side: `{config.get('max_side', '-')}`",
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
        description="One-shot instance detection benchmark and matcher validation."
    )
    parser.add_argument(
        "--protocol",
        choices=["box", "hpatches", "retrieval"],
        default="box",
        help=(
            "Benchmark protocol: box AP, native HPatches metrics, or Paris6k retrieval."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON or JSONL manifest. Required for the box protocol.",
    )
    parser.add_argument("--root", type=Path, help="Base directory for manifest paths.")
    parser.add_argument(
        "--hpatches-dir",
        type=Path,
        default=Path("data/hpatches"),
        help="Directory containing hpatches-sequences-release.",
    )
    parser.add_argument(
        "--paris-dir",
        type=Path,
        default=Path("data/paris6k"),
        help="Directory containing the Paris6k manifest and images.",
    )
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
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit samples (box), pairs (HPatches), or queries (retrieval).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for a hit in the box protocol.",
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
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help=(
            "Longest image side fed to the dense matchers (LoFTR, FlexGlue); "
            "larger images are downscaled and keypoints mapped back."
        ),
    )
    parser.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=0.0,
        help=(
            "Cap this process to a fraction of total VRAM; 0 auto-sizes from "
            "currently free memory."
        ),
    )
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
    names = _selected_models(args.models)
    _configure_gpu(args.gpu_mem_fraction, log)

    if args.protocol == "hpatches":
        reports, config = _run_hpatches(args, names, log)
        json_path, markdown_path = write_hpatches_reports(reports, config, args.output)
    elif args.protocol == "retrieval":
        reports, config = _run_retrieval(args, names, log)
        json_path, markdown_path = write_retrieval_reports(reports, config, args.output)
    else:
        if args.manifest is None:
            raise SystemExit("error: --manifest is required for the box protocol")
        reports, config = _run_box(args, names, log)
        json_path, markdown_path = write_reports(reports, config, args.output)

    log.info("wrote %s", json_path)
    log.info("wrote %s", markdown_path)
    return 0


def _run_box(
    args: argparse.Namespace,
    names: Sequence[str],
    log: Logger,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the box-AP benchmark.

    Args:
        args: Parsed arguments.
        names: Model names.
        log: Logger.

    Returns:
        Reports and config.
    """
    dataset = InstanceDataset(args.manifest, root=args.root)
    samples = dataset.samples
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    config: dict[str, object] = {
        "protocol": "box",
        "manifest": str(args.manifest),
        "models": list(names),
        "samples": len(samples),
        "iou_threshold": args.iou_threshold,
        "max_keypoints": args.max_keypoints,
        "ratio": args.ratio,
        "ransac_threshold": args.ransac_threshold,
        "max_side": args.max_side,
    }
    reports: list[dict[str, object]] = []
    for name in names:
        matcher: FeatureMatcher | None = None
        _reset_gpu_peak()
        try:
            matcher = _make_matcher(name, args)
            report = run_benchmark(
                matcher,
                samples,
                args.iou_threshold,
                loader=dataset.load,
            )
            report.update(_gpu_memory_stats())
            reports.append(report)
            latency = report["latency_ms"]
            log.info(
                "%s: hit=%.3f ap50=%.3f iou=%.3f p50=%.1fms peak=%.0fMiB",
                name,
                report["hit_rate"],
                report["ap50"],
                report["mean_iou"],
                latency["p50"],
                report.get("peak_gpu_mib", 0.0),
            )
        except torch.OutOfMemoryError as error:
            log.error(
                "%s: skipped, GPU out of memory: %s",
                name,
                str(error).splitlines()[0],
            )
        finally:
            matcher = None
            _release_gpu()
    return reports, config


def _run_hpatches(
    args: argparse.Namespace,
    names: Sequence[str],
    log: Logger,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run native HPatches validation.

    Args:
        args: Parsed arguments.
        names: Model names.
        log: Logger.

    Returns:
        Reports and config.
    """
    sequences_dir = args.hpatches_dir / "hpatches-sequences-release"
    if not sequences_dir.exists():
        raise SystemExit(
            f"missing HPatches sequences: {sequences_dir}\n"
            "run: uv run python scripts/prepare_hpatches.py"
        )
    pairs = load_pairs(sequences_dir, limit=args.max_samples)
    if not pairs:
        raise SystemExit(f"no HPatches pairs found under {sequences_dir}")

    config: dict[str, object] = {
        "protocol": "hpatches",
        "sequences_dir": str(sequences_dir),
        "sequences": len({pair.sequence for pair in pairs}),
        "pairs": len(pairs),
        "models": list(names),
        "max_keypoints": args.max_keypoints,
        "ratio": args.ratio,
        "ransac_threshold": args.ransac_threshold,
        "max_side": args.max_side,
    }
    reports: list[dict[str, object]] = []
    for name in names:
        matcher: FeatureMatcher | None = None
        _reset_gpu_peak()
        try:
            matcher = _make_matcher(name, args)
            report = evaluate_hpatches(matcher, pairs)
            report.update(_gpu_memory_stats())
            reports.append(report)
            log.info(
                "%s: MMA@3=%s HEA@3=%s median_CE=%s peak=%.0fMiB",
                name,
                report["mma"]["3"],
                report["hea"]["3"],
                report["corner_error"]["median"],
                report.get("peak_gpu_mib", 0.0),
            )
        except torch.OutOfMemoryError as error:
            log.error(
                "%s: skipped, GPU out of memory: %s",
                name,
                str(error).splitlines()[0],
            )
        finally:
            matcher = None
            _release_gpu()
    return reports, config


def _run_retrieval(
    args: argparse.Namespace,
    names: Sequence[str],
    log: Logger,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run Paris6k retrieval evaluation.

    Args:
        args: Parsed arguments.
        names: Model names.
        log: Logger.

    Returns:
        Reports and config.
    """
    manifest = args.paris_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(
            f"missing Paris6k manifest: {manifest}\n"
            "run: uv run python scripts/prepare_paris6k.py"
        )
    queries = load_queries(manifest, limit=args.max_samples)
    if not queries:
        raise SystemExit(f"no Paris6k queries found in {manifest}")

    candidate_counts = [len(query.candidates) for query in queries]
    config: dict[str, object] = {
        "protocol": "retrieval",
        "manifest": str(manifest),
        "queries": len(queries),
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
        "models": list(names),
        "max_keypoints": args.max_keypoints,
        "ratio": args.ratio,
        "ransac_threshold": args.ransac_threshold,
        "max_side": args.max_side,
    }
    reports: list[dict[str, object]] = []
    for name in names:
        matcher: FeatureMatcher | None = None
        _reset_gpu_peak()
        try:
            matcher = _make_matcher(name, args)
            report = evaluate_retrieval(matcher, queries)
            report.update(_gpu_memory_stats())
            reports.append(report)
            log.info(
                "%s: mAP=%s R@1=%s R@5=%s peak=%.0fMiB",
                name,
                report["map"],
                report["recall"]["1"],
                report["recall"]["5"],
                report.get("peak_gpu_mib", 0.0),
            )
        except torch.OutOfMemoryError as error:
            log.error(
                "%s: skipped, GPU out of memory: %s",
                name,
                str(error).splitlines()[0],
            )
        finally:
            matcher = None
            _release_gpu()
    return reports, config


def _make_matcher(name: str, args: argparse.Namespace) -> FeatureMatcher:
    """Construct a matcher from CLI settings.

    Args:
        name: Model name.
        args: Parsed arguments.

    Returns:
        A configured matcher.
    """
    kwargs: dict[str, object] = {
        "max_keypoints": args.max_keypoints,
        "ratio": args.ratio,
        "ransac_threshold": args.ransac_threshold,
    }
    if name in {"loftr", "flexglue"}:
        kwargs["max_side"] = args.max_side
    return MODELS[name](**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
