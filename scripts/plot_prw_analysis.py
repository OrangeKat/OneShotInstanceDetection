#!/usr/bin/env python3
"""Plot the PRW complexity analysis for the devlog."""

from __future__ import annotations

# ruff: noqa: E402
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.plots.common import plt, save_figure

DATASETS = ["PRW", "Flickr", "RPC"]
MANIFESTS = {
    "PRW": Path("data/prw/manifest-q200.json"),
    "Flickr": Path("data/flickrlogos32/manifest.json"),
    "RPC": Path("data/rpc/manifest.json"),
}
MODELS = [
    "sift",
    "superpoint_lightglue",
    "aliked_lightglue",
    "loftr",
    "flexglue",
]
PRETTY = {
    "sift": "SIFT",
    "orb": "ORB",
    "akaze": "AKAZE",
    "superpoint_lightglue": "SuperPoint+LG",
    "aliked_lightglue": "ALIKED+LG",
    "loftr": "LoFTR",
    "flexglue": "FlexGlue",
    "roma": "RoMa",
}
COLORS = {
    "no_matches": "#8c2d04",
    "too_few_matches": "#e6550d",
    "ransac_failed": "#a63603",
    "degenerate_homography": "#6a51a3",
    "wrong_box": "#fec44f",
    "hit": "#31a354",
    "false_positive": "#bcbddc",
    "true_negative": "#d9d9d9",
}
STAGE_ORDER = [
    "no_matches",
    "too_few_matches",
    "ransac_failed",
    "degenerate_homography",
    "wrong_box",
    "hit",
    "false_positive",
    "true_negative",
]


def main() -> int:
    """Generate the PRW figures.

    Returns:
        Process exit code.
    """
    output = Path("docs/devnotes/images/week2")
    output.mkdir(parents=True, exist_ok=True)

    area = {name: area_stats(path) for name, path in MANIFESTS.items()}
    plot_area(area, output / "prw_area_comparison.png")

    reports = load_reports()
    plot_matches(reports, output / "prw_matches_comparison.png")
    plot_stages(reports, output / "prw_stages.png")
    print(f"wrote figures to {output}")
    return 0


def area_stats(path: Path) -> dict[str, float]:
    """Compute GT box area statistics for positive pairs.

    Args:
        path: Manifest path.

    Returns:
        Area percentages and reference/GT area ratio statistics.
    """
    manifest = json.loads(path.read_text())
    root = path.parent
    areas: list[float] = []
    ratios: list[float] = []
    ref_cache: dict[Path, tuple[int, int] | None] = {}
    for record in manifest:
        if not record["query_boxes"]:
            continue
        image = cv2.imread(str(root / record["query"]))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = max(
            record["query_boxes"],
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        )
        gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
        areas.append(100.0 * gt_area / max(1.0, height * width))
        reference = root / record["reference"]
        if reference not in ref_cache:
            ref = cv2.imread(str(reference))
            ref_cache[reference] = None if ref is None else ref.shape[:2]
        if ref_cache[reference] is not None:
            ref_height, ref_width = ref_cache[reference]
            ratios.append(ref_width * ref_height / max(1.0, gt_area))
    array = np.asarray(areas)
    return {
        "median_area": float(np.median(array)),
        "below_1": float((array < 1.0).mean() * 100.0),
        "below_2": float((array < 2.0).mean() * 100.0),
        "ref_ratio": float(statistics.median(ratios)),
    }


def load_reports() -> dict[str, dict[str, object]]:
    """Merge classical and learned PRW reports.

    Returns:
        Model name to report.
    """
    reports: dict[str, dict[str, object]] = {}
    for path in (
        Path("outputs/prw-q200/results.json"),
        Path("outputs/prw-q200-learned/results.json"),
        Path("outputs/prw-q200-roma/results.json"),
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        reports.update({str(report["model"]): report for report in payload["models"]})
    return reports


def plot_area(area: dict[str, dict[str, float]], path: Path) -> None:
    """Plot object-area comparison.

    Args:
        area: Dataset statistics.
        path: Output path.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(DATASETS))
    left.bar(x, [area[name]["median_area"] for name in DATASETS], color="#4c72b0")
    left.set_xticks(x)
    left.set_xticklabels(DATASETS)
    left.set_ylabel("Median GT box area (% of query)")
    left.set_title("Instance size")
    for index, name in enumerate(DATASETS):
        left.text(
            index,
            area[name]["median_area"] + 0.15,
            f"{area[name]['median_area']:.2f}%",
            ha="center",
            fontsize=8,
        )

    width = 0.38
    right.bar(
        x - width / 2,
        [area[name]["below_1"] for name in DATASETS],
        width,
        label="< 1%",
        color="#dd8452",
    )
    right.bar(
        x + width / 2,
        [area[name]["below_2"] for name in DATASETS],
        width,
        label="< 2%",
        color="#55a868",
    )
    right.set_xticks(x)
    right.set_xticklabels(DATASETS)
    right.set_ylim(0, 100)
    right.set_ylabel("Positive pairs (%)")
    right.set_title("Small instances")
    right.legend(frameon=False)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path)


def plot_matches(reports: dict[str, dict[str, object]], path: Path) -> None:
    """Plot mean matches across datasets.

    Args:
        reports: Merged PRW reports.
        path: Output path.
    """
    sources = {
        "PRW": reports,
        "Flickr": read_models(Path("outputs/flickrlogos32/results.json")),
        "RPC": read_models(Path("outputs/rpc/results.json")),
    }
    x = np.arange(len(MODELS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for index, (name, models) in enumerate(sources.items()):
        values = [float(models[model]["mean_matches"]) for model in MODELS]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=name,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.05,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[model] for model in MODELS], rotation=15)
    ax.set_ylabel("Mean matches per pair (log scale)")
    ax.set_title("Matches collapse on PRW")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path)


def plot_stages(reports: dict[str, dict[str, object]], path: Path) -> None:
    """Plot stage distribution per PRW model.

    Args:
        reports: Merged PRW reports.
        path: Output path.
    """
    models = [model for model in PRETTY if model in reports]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    left = np.zeros(len(models))
    for stage in STAGE_ORDER:
        values = []
        for model in models:
            counts = reports[model].get("stages", {})
            total = max(1, sum(counts.values()))
            values.append(100.0 * counts.get(stage, 0) / total)
        ax.barh(
            models,
            values,
            left=left,
            color=COLORS[stage],
            label=stage.replace("_", " "),
        )
        left += np.asarray(values)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pairs (%)")
    ax.set_title("PRW failure stages")
    ax.legend(
        ncol=2,
        fontsize=7,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([PRETTY[model] for model in models])
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path)


def read_models(path: Path) -> dict[str, dict[str, object]]:
    """Read a results file into a model map.

    Args:
        path: Results path.

    Returns:
        Model name to report.
    """
    payload = json.loads(path.read_text())
    return {str(report["model"]): report for report in payload["models"]}


if __name__ == "__main__":
    raise SystemExit(main())
