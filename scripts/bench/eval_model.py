#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a real dataset subset.

Runs the model's benchmark adapter over a sampled slice of a manifest and
compares it with the baseline matchers on the same pairs.

Usage:
    uv run python scripts/bench/eval_model.py \
        --checkpoint outputs/train/heads.pt --samples 200 --baselines sift
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from osid.benchmark.registry import MODELS  # noqa: E402
from osid.benchmark.runner import run_benchmark  # noqa: E402
from osid.model.adapter import BenchmarkAdapter  # noqa: E402
from osid.model.train import load_checkpoint  # noqa: E402
from osid.utils.dataset import InstanceDataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--baselines", nargs="*", default=["sift"])
    parser.add_argument("--matching", choices=("mutual", "many"), default="mutual")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    dataset = InstanceDataset(args.manifest)
    rng = np.random.default_rng(args.seed)
    positives = [s for s in dataset.samples if s.query_boxes]
    negatives = [s for s in dataset.samples if not s.query_boxes]
    chosen = [
        positives[i] for i in rng.choice(len(positives), args.samples, replace=False)
    ]
    if negatives:
        chosen += [
            negatives[i]
            for i in rng.choice(
                len(negatives), min(args.negatives, len(negatives)), replace=False
            )
        ]
    print(
        f"pairs: {len(chosen)} ({args.samples} positive, {len(chosen) - args.samples} negative)"
    )

    adapter = BenchmarkAdapter(
        load_checkpoint(args.checkpoint, device=args.device),
        matching=args.matching,
    )
    reports = {
        "osid_boundary": run_benchmark(
            adapter, chosen, args.iou_threshold, loader=dataset.load
        )
    }
    for name in args.baselines:
        matcher = MODELS[name](max_keypoints=adapter.detector.config.top_k)
        reports[name] = run_benchmark(
            matcher, chosen, args.iou_threshold, loader=dataset.load
        )

    print(
        f"{'model':16s} {'hit':>6s} {'AP@.5':>6s} {'IoU':>6s} "
        f"{'matches':>8s} {'inliers':>8s} {'p50 ms':>7s}"
    )
    for name, report in reports.items():
        latency = report["latency_ms"]["p50"]
        print(
            f"{name:16s} {report['hit_rate']:6.3f} {report['ap50']:6.3f} "
            f"{report['mean_iou']:6.3f} {report['mean_matches']:8.1f} "
            f"{report['mean_inliers']:8.1f} {latency:7.0f}"
        )

    print("\nstages:")
    for name, report in reports.items():
        counts = Counter(sample["stage"] for sample in report["per_sample"])
        total = sum(counts.values())
        cells = ", ".join(
            f"{stage} {count} ({count / total:.0%})"
            for stage, count in counts.most_common(4)
        )
        print(f"  {name:16s} {cells}")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        path = args.output / f"{name}.json"
        path.write_text(
            json.dumps(report, indent=2)[:0]
            + json.dumps(
                {
                    "config": {
                        "manifest": str(args.manifest),
                        "samples": len(chosen),
                        "iou_threshold": args.iou_threshold,
                    },
                    "models": [{**report, "model": name}],
                },
                indent=2,
            )
            + "\n"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
