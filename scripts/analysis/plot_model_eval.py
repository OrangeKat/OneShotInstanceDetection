#!/usr/bin/env python3
"""Plot the trained model's results: training curves and real-data comparison.

Reads the training history and the evaluation JSON written by
``scripts/bench/eval_model.py``, and writes one figure with two panels.

Usage:
    uv run python scripts/analysis/plot_model_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HISTORY = Path("outputs/train/history.json")
EVAL_DIR = Path("outputs/eval")
OUTPUT = Path("docs/devnotes/images/week4/model_eval.png")


def main() -> int:
    """Render the figure.

    Returns:
        Process exit code.
    """
    history = json.loads(HISTORY.read_text())
    steps = [row["step"] for row in history]
    figure, (curves, bars) = plt.subplots(1, 2, figsize=(10.4, 3.9))

    curves.plot(
        steps, [row["heat"] for row in history], marker="o", ms=3, label="boundary loss"
    )
    curves.plot(
        steps,
        [row["descriptor"] for row in history],
        marker="o",
        ms=3,
        label="descriptor loss",
    )
    curves.axhline(np.log(128), ls="--", lw=1, color="#888", label="chance (ln 128)")
    curves.set_xlabel("step")
    curves.set_ylabel("loss")
    curves.set_yscale("log")
    curves.set_title("Training, synthetic shapes")
    curves.legend(frameon=False, fontsize=8)

    scores = {
        "osid_boundary": json.loads((EVAL_DIR / "osid_boundary.json").read_text())[
            "models"
        ][0],
        "sift": json.loads((EVAL_DIR / "sift.json").read_text())["models"][0],
    }
    names = list(scores)
    positions = np.arange(len(names))
    width = 0.36
    bars.bar(
        positions - width / 2,
        [scores[n]["hit_rate"] for n in names],
        width,
        label="hit rate",
        color="#5aa9ff",
    )
    bars.bar(
        positions + width / 2,
        [scores[n]["ap50"] for n in names],
        width,
        label="AP@0.5",
        color="#35c07a",
    )
    bars.set_xticks(positions)
    bars.set_xticklabels(names)
    bars.set_ylim(0, 0.5)
    bars.set_ylabel("score")
    bars.set_title("FlickrLogos-32, 150 positives + 75 negatives")
    for index, name in enumerate(names):
        for offset, key in ((-width / 2, "hit_rate"), (width / 2, "ap50")):
            value = scores[name][key]
            bars.text(
                index + offset, value + 0.01, f"{value:.3f}", ha="center", fontsize=8
            )
    bars.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=160)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
