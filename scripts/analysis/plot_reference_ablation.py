#!/usr/bin/env python3
"""Plot the reference-quality ablation: default crops vs chosen crops.

Reads the two FlickrLogos-32 result files and writes one paired bar chart of hit
rate per model, plus the hard-pair counts.

Usage:
    uv run python scripts/analysis/plot_reference_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_RUN = Path("outputs/flickrlogos32/results.json")
CHOSEN_RUN = Path("outputs/flickrlogos32-refs/results.json")
OUTPUT = Path("docs/devnotes/images/week3/reference_ablation.png")
LABELS = {
    "sift": "SIFT",
    "orb": "ORB",
    "akaze": "AKAZE",
    "superpoint_lightglue": "SP + LG",
    "aliked_lightglue": "ALIKED + LG",
    "loftr": "LoFTR",
    "roma": "RoMa",
    "flexglue": "FlexGlue",
}


def read(path: Path) -> dict[str, dict[str, float]]:
    """Return hit rates per model from a results file.

    Args:
        path: Results JSON.

    Returns:
        Model to metrics mapping.
    """
    payload = json.loads(path.read_text())
    return {
        str(report["model"]): {
            "hit": float(report["hit_rate"]),
            "ap": float(report["ap50"]),
        }
        for report in payload["models"]
    }


def main() -> int:
    """Render the comparison chart.

    Returns:
        Process exit code.
    """
    default = read(DEFAULT_RUN)
    chosen = read(CHOSEN_RUN)
    models = sorted(default, key=lambda name: default[name]["hit"])

    figure, (axis, counts) = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.8),
        gridspec_kw={"width_ratios": [3.4, 1]},
    )
    positions = np.arange(len(models))
    width = 0.38
    axis.bar(
        positions - width / 2,
        [default[name]["hit"] for name in models],
        width,
        label="default crops",
        color="#5aa9ff",
    )
    axis.bar(
        positions + width / 2,
        [chosen[name]["hit"] for name in models],
        width,
        label="chosen crops",
        color="#35c07a",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [LABELS.get(name, name) for name in models], rotation=25, ha="right"
    )
    axis.set_ylabel("Hit rate at IoU ≥ 0.5")
    axis.set_ylim(0, 1)
    axis.set_title("FlickrLogos-32: reference crops changed, protocol held fixed")
    axis.legend(frameon=False, fontsize=8)

    counts.bar([0], [381], color="#5aa9ff", width=0.6, label="default")
    counts.bar([1], [251], color="#35c07a", width=0.6, label="chosen")
    counts.set_xticks([0, 1])
    counts.set_xticklabels(["default", "chosen"])
    counts.set_ylabel("Pairs no model solves")
    counts.set_title("Hard pairs")
    for index, value in enumerate((381, 251)):
        counts.text(index, value + 6, str(value), ha="center", fontsize=9)

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=160)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
