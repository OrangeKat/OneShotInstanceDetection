"""Command-line entry point for the benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from logging import Logger
from pathlib import Path

import torch

from osid.benchmark.gpu import (
    configure_gpu,
    gpu_memory_stats,
    release_gpu,
    reset_gpu_peak,
)
from osid.benchmark.matchers.base import FeatureMatcher
from osid.benchmark.registry import MODELS
from osid.benchmark.reports import write_hpatches_reports, write_reports
from osid.benchmark.runner import run_benchmark
from osid.benchmark.validation.hpatches import evaluate_model as evaluate_hpatches
from osid.benchmark.validation.hpatches import load_pairs
from osid.utils.dataset import InstanceDataset
from osid.utils.logger import ProgressReporter, get_logger
from osid.utils.model_info import parameter_count
from osid.utils.seeding import set_seed


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
        choices=["box", "hpatches"],
        default="box",
        help="Benchmark protocol: box AP or native HPatches metrics.",
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
        help="Limit samples (box) or pairs (HPatches).",
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
        "--roma-coarse-res",
        type=int,
        default=560,
        help="RoMa coarse resolution; lower values are faster.",
    )
    parser.add_argument(
        "--roma-upsample-res",
        type=int,
        default=864,
        help="RoMa upsample resolution; lower values are faster.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for Python, NumPy, PyTorch, and OpenCV RANSAC.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Log progress every N items; 0 auto-sizes to about 20 updates.",
    )
    return parser.parse_args(argv)


def selected_models(names: Sequence[str]) -> list[str]:
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
    names = selected_models(args.models)
    set_seed(args.seed)
    log.info(
        "benchmark | protocol=%s | models=%s | output=%s",
        args.protocol,
        ", ".join(names),
        args.output,
    )
    configure_gpu(args.gpu_mem_fraction, log)

    if args.protocol == "hpatches":
        reports, config = run_hpatches(args, names, log)
        json_path, markdown_path = write_hpatches_reports(reports, config, args.output)
    else:
        if args.manifest is None:
            raise SystemExit("error: --manifest is required for the box protocol")
        reports, config = run_box(args, names, log)
        json_path, markdown_path = write_reports(reports, config, args.output)

    log.info("wrote %s", json_path)
    log.info("wrote %s", markdown_path)
    return 0


def run_box(
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
        "roma_coarse_res": args.roma_coarse_res,
        "roma_upsample_res": args.roma_upsample_res,
        "seed": args.seed,
    }
    log.info("box | %d samples | manifest=%s", len(samples), args.manifest)
    log.info(
        "config | keypoints=%d ratio=%.2f ransac=%.1f max_side=%d seed=%d",
        args.max_keypoints,
        args.ratio,
        args.ransac_threshold,
        args.max_side,
        args.seed,
    )
    reports: list[dict[str, object]] = []
    for model_index, name in enumerate(names, start=1):
        matcher: FeatureMatcher | None = None
        reset_gpu_peak()
        log.info("model %d/%d: %s", model_index, len(names), name)
        try:
            matcher = make_matcher(name, args)
            reporter = ProgressReporter(
                log,
                name,
                len(samples),
                args.progress_every,
            )
            report = run_benchmark(
                matcher,
                samples,
                args.iou_threshold,
                loader=dataset.load,
                progress=reporter.update,
            )
            report.update(gpu_memory_stats())
            report["params"] = parameter_count(matcher)
            reports.append(report)
            write_reports(reports, config, args.output)
            log.info(
                "checkpoint | %d/%d models -> %s",
                len(reports),
                len(names),
                args.output,
            )
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
            release_gpu()
    log.info("box complete | %d/%d models | %s", len(reports), len(names), args.output)
    return reports, config


def run_hpatches(
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
        "roma_coarse_res": args.roma_coarse_res,
        "roma_upsample_res": args.roma_upsample_res,
        "seed": args.seed,
    }
    log.info(
        "hpatches | %d pairs | %d sequences | %s",
        len(pairs),
        len({pair.sequence for pair in pairs}),
        sequences_dir,
    )
    reports: list[dict[str, object]] = []
    for model_index, name in enumerate(names, start=1):
        matcher: FeatureMatcher | None = None
        reset_gpu_peak()
        log.info("model %d/%d: %s", model_index, len(names), name)
        try:
            matcher = make_matcher(name, args)
            reporter = ProgressReporter(log, name, len(pairs), args.progress_every)
            report = evaluate_hpatches(matcher, pairs, progress=reporter.update)
            report.update(gpu_memory_stats())
            report["params"] = parameter_count(matcher)
            reports.append(report)
            write_hpatches_reports(reports, config, args.output)
            log.info(
                "checkpoint | %d/%d models -> %s",
                len(reports),
                len(names),
                args.output,
            )
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
            release_gpu()
    log.info(
        "hpatches complete | %d/%d models | %s",
        len(reports),
        len(names),
        args.output,
    )
    return reports, config


def make_matcher(name: str, args: argparse.Namespace) -> FeatureMatcher:
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
    if name == "roma":
        kwargs["coarse_res"] = args.roma_coarse_res
        kwargs["upsample_res"] = args.roma_upsample_res
    return MODELS[name](**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
