"""Markdown report for HPatches matcher validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

THRESHOLDS = (1, 3, 5, 10)
SPLIT_THRESHOLDS = (3, 5, 10)
LITERATURE = {
    "sift": {
        "mma": [0.60, 0.87, 0.93, 0.95],
        "hea": [0.51, 0.76, 0.86, 0.91],
    },
    "orb": {
        "mma": [0.37, 0.68, 0.73, 0.74],
        "hea": [0.13, 0.40, 0.51, 0.62],
    },
    "akaze": {
        "mma": [0.50, 0.80, 0.87, 0.89],
        "hea": [0.34, 0.65, 0.75, 0.83],
    },
}


def render_report(
    results: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> str:
    """Render the HPatches validation report.

    Args:
        results: Per-model metric dictionaries.
        config: Run configuration.

    Returns:
        Markdown document.
    """
    mma_headers = " | ".join(f"MMA@{threshold}" for threshold in THRESHOLDS)
    hea_headers = " | ".join(f"HEA@{threshold}" for threshold in THRESHOLDS)
    lines = [
        "# HPatches matcher validation",
        "",
        f"- Data: `{config['sequences_dir']}`",
        f"- Sequences: `{config['sequences']}`",
        f"- Pairs: `{config['pairs']}`",
        f"- Max keypoints: `{config['max_keypoints']}`",
        f"- Lowe ratio: `{config['ratio']}`",
        f"- RANSAC threshold: `{config['ransac_threshold']}` px",
        "",
        "## Metrics",
        "",
        "| Model | Pairs | With matches | With H | "
        + mma_headers
        + " | "
        + hea_headers
        + " | Median CE |",
        "|---|---:|---:|---:|" + "---:|" * (len(THRESHOLDS) * 2) + "---:|",
    ]
    for result in results:
        mma = result["mma"]
        hea = result["hea"]
        corner = result["corner_error"]
        lines.append(
            f"| {result['model']} | {result['pairs']} "
            f"| {result['pairs_with_matches']} "
            f"| {result['pairs_with_homography']} | "
            + " | ".join(_fmt(mma[str(threshold)]) for threshold in THRESHOLDS)
            + " | "
            + " | ".join(_fmt(hea[str(threshold)]) for threshold in THRESHOLDS)
            + f" | {_fmt(corner['median'])} |"
        )

    lines.extend(
        [
            "",
            "## Literature reference",
            "",
            "Efe et al. 2021, Table 2 — OpenCV 4.5.2, HPatches sequences,",
            "tuned ratio test + mutual nearest neighbors, 108 sequences.",
            "",
            "| Model | MMA@1 | MMA@3 | MMA@5 | MMA@10 "
            "| HEA@1 | HEA@3 | HEA@5 | HEA@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, values in LITERATURE.items():
        lines.append(
            f"| {model} | "
            + " | ".join(f"{value:.2f}" for value in values["mma"])
            + " | "
            + " | ".join(f"{value:.2f}" for value in values["hea"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per split",
            "",
            "| Model | Split | Pairs | MMA@3 | MMA@5 | MMA@10 "
            "| HEA@3 | HEA@5 | HEA@10 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        for split in ("illumination", "viewpoint"):
            subset = [pair for pair in result["per_pair"] if pair["split"] == split]
            if not subset:
                continue
            mma_cells = []
            hea_cells = []
            for threshold in SPLIT_THRESHOLDS:
                mma = [
                    pair["mma"][str(threshold)]
                    for pair in subset
                    if pair["mma"][str(threshold)] is not None
                ]
                mma_cells.append(_fmt(float(np.mean(mma)) if mma else None))
                hea = [
                    pair["corner_error"] is not None
                    and pair["corner_error"] <= threshold
                    for pair in subset
                ]
                hea_cells.append(_fmt(float(np.mean(hea)) if hea else None))
            lines.append(
                f"| {result['model']} | {split} | {len(subset)} | "
                + " | ".join(mma_cells)
                + " | "
                + " | ".join(hea_cells)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            "- MMA: per-pair fraction of matches within the pixel threshold after "
            "projecting reference keypoints through the ground-truth homography; "
            "averaged over pairs with at least one match.",
            "- HEA: fraction of all pairs whose mean four-corner error is within "
            "the pixel threshold; missing homographies count as failures.",
            "- Corner error: mean Euclidean distance between the four reference "
            "corners projected by the ground-truth and estimated homographies.",
            "- This run uses all 116 sequences, a simple ratio test at 0.75, and "
            "2,048 keypoints. The literature reference uses 108 sequences, "
            "mutual nearest neighbors with a bidirectional ratio test, tuned "
            "thresholds, and OpenCV 4.5.2. Expect differences.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    """Format an optional metric value.

    Args:
        value: Metric value or ``None``.

    Returns:
        Formatted string.
    """
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
