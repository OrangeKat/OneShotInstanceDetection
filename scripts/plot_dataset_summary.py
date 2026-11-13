#!/usr/bin/env python3
"""Generate the Week 2 per-dataset result figures for the devlog."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.data import LOGO_SIZE_BINS, LOGO_SIZE_LABELS, Row, load_rows
from src.analysis.metrics import binned_hit_rates
from src.analysis.plots.common import plt, save_figure

PRETTY = {
    "sift": "SIFT",
    "orb": "ORB",
    "akaze": "AKAZE",
    "superpoint_lightglue": "SuperPoint+LG",
    "aliked_lightglue": "ALIKED+LG",
    "loftr": "LoFTR",
    "roma": "RoMa",
    "flexglue": "FlexGlue",
}
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Write all figures.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Plot per-dataset results.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/devnotes/images/week2"),
        help="Figure output directory.",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    flickr = read_payload(Path("outputs/flickrlogos32/results.json"))
    rpc = read_payload(Path("outputs/rpc/results.json"))
    hpatches = read_payload(Path("outputs/hpatches/results.json"))
    hpatches.update(read_payload(Path("outputs/hpatches-roma-2048/results.json")))
    prw: dict[str, dict[str, object]] = {}
    for path in (
        Path("outputs/prw-q200/results.json"),
        Path("outputs/prw-q200-learned/results.json"),
        Path("outputs/prw-q200-roma/results.json"),
    ):
        prw.update(read_payload(path))

    plot_models(
        flickr,
        ("hit_rate", "ap50"),
        ("Hit rate", "AP@0.5"),
        "FlickrLogos-32: instance-level logos",
        args.output / "flickr_models.png",
    )
    plot_models(
        rpc,
        ("hit_rate", "ap50"),
        ("Hit rate", "AP@0.5"),
        "RPC: category-level products",
        args.output / "rpc_models.png",
    )
    plot_models(
        hpatches,
        ("mma3", "hea3"),
        ("MMA@3", "HEA@3"),
        "HPatches: matcher validation",
        args.output / "hpatches_models.png",
    )
    plot_matrix(flickr, rpc, hpatches, prw, args.output / "dataset_matrix.png")

    flickr_rows, _ = load_rows(
        Path("outputs/flickrlogos32/results.json"),
        Path("data/flickrlogos32/manifest.json"),
    )
    plot_area(
        flickr_rows,
        ["roma", "flexglue", "loftr", "superpoint_lightglue", "sift"],
        "FlickrLogos-32: accuracy by logo area",
        args.output / "flickr_accuracy_by_area.png",
    )
    rpc_rows, _ = load_rows(
        Path("outputs/rpc/results.json"),
        Path("data/rpc/manifest.json"),
    )
    plot_area(
        rpc_rows,
        ["flexglue", "sift", "roma", "akaze", "orb"],
        "RPC: accuracy by product area",
        args.output / "rpc_accuracy_by_area.png",
    )
    print(f"wrote figures to {args.output}")
    return 0


def read_payload(path: Path) -> dict[str, dict[str, object]]:
    """Read a results JSON into a model map.

    Args:
        path: Results path.

    Returns:
        Model name to report, plus derived metrics.
    """
    payload = json.loads(path.read_text())
    reports: dict[str, dict[str, object]] = {}
    for report in payload["models"]:
        model = str(report["model"])
        row = dict(report)
        if "mma" in row:
            row["mma3"] = row["mma"]["3"]
            row["hea3"] = row["hea"]["3"]
        reports[model] = row
    return reports


def plot_models(
    reports: Mapping[str, Mapping[str, object]],
    metrics: tuple[str, str],
    labels: tuple[str, str],
    title: str,
    path: Path,
) -> None:
    """Plot grouped bars for two metrics.

    Args:
        reports: Model reports.
        metrics: Metric keys.
        labels: Metric labels.
        title: Figure title.
        path: Output path.
    """
    models = sorted(
        reports,
        key=lambda model: float(reports[model].get(metrics[1], 0.0)),
        reverse=True,
    )
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        values = [float(reports[model].get(metric, 0.0)) for model in models]
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=("#4c72b0", "#dd8452")[index],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(model, model) for model in models], rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path)


def plot_area(
    rows: Sequence[Row],
    models: Sequence[str],
    title: str,
    path: Path,
) -> None:
    """Plot hit rate over object-area buckets.

    Args:
        rows: Enriched rows.
        models: Model names.
        title: Figure title.
        path: Output path.
    """
    data = binned_hit_rates(
        rows,
        models,
        "logo_area_pct",
        LOGO_SIZE_BINS,
        LOGO_SIZE_LABELS,
    )
    labels = [
        f"{label}\n(n={data[models[0]][0][index]})"
        for index, label in enumerate(LOGO_SIZE_LABELS)
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for index, model in enumerate(models):
        _, rates = data[model]
        values = [np.nan if np.isnan(rate) else rate for rate in rates]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            color=COLORS[index % len(COLORS)],
            label=PRETTY.get(model, model),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Hit rate")
    ax.set_xlabel("Largest GT box as % of query image")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path)


def plot_matrix(
    flickr: Mapping[str, Mapping[str, object]],
    rpc: Mapping[str, Mapping[str, object]],
    hpatches: Mapping[str, Mapping[str, object]],
    prw: Mapping[str, Mapping[str, object]],
    path: Path,
) -> None:
    """Plot a model x dataset primary-metric heatmap.

    Args:
        flickr: Flickr reports.
        rpc: RPC reports.
        hpatches: HPatches reports.
        prw: PRW reports.
        path: Output path.
    """
    columns = [
        (
            "Flickr AP@0.5",
            {model: float(row.get("ap50", 0.0)) for model, row in flickr.items()},
        ),
        (
            "RPC AP@0.5",
            {model: float(row.get("ap50", 0.0)) for model, row in rpc.items()},
        ),
        (
            "HPatches HEA@3",
            {model: float(row.get("hea3", 0.0)) for model, row in hpatches.items()},
        ),
        (
            "PRW AP@0.5 (200 q)",
            {model: float(row.get("ap50", 0.0)) for model, row in prw.items()},
        ),
    ]
    models = sorted(
        set(flickr) | set(rpc) | set(hpatches) | set(prw),
        key=lambda model: (
            -float(flickr.get(model, {}).get("ap50", 0.0)),
            PRETTY.get(model, model),
        ),
    )
    matrix = np.asarray(
        [[values.get(model, np.nan) for _, values in columns] for model in models]
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([label for label, _ in columns], rotation=15)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([PRETTY.get(model, model) for model in models])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "n/a" if np.isnan(value) else f"{value:.2f}"
            color = "white" if not np.isnan(value) and value > 0.4 else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Primary metric by dataset")
    save_figure(fig, path)


if __name__ == "__main__":
    raise SystemExit(main())
