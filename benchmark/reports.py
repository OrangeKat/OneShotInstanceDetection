"""Markdown and JSON report writers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from benchmark.validation.report import render_report


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
    markdown_path.write_text(markdown(reports, config))
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


def markdown(
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
