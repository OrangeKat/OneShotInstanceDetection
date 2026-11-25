"""Failure taxonomy: assign each failed pair a cause from recorded evidence.

The stage breakdown says where a pair died. The taxonomy says why, using rules
over the diagnostics the benchmark already records, so the labels are
reproducible instead of judgement calls.

Causes (mutually exclusive, in precedence order):

``no_signal``
    Fewer than four matches survived filtering, so no geometry could be
    estimated. The image pair carries too little usable signal.
``ambiguous``
    Matches exist and a box was produced, but it lands off the instance
    (IoU < 0.1), or matches existed without a consistent homography. The matcher
    locked onto the background or another object.
``weak_geometry``
    A box landed on the instance but too loosely (0.1 <= IoU < 0.5), or the
    homography was degenerate. The correspondence set is right; the estimate is
    not.
``hit``
    IoU >= 0.5.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Final

from src.analysis.data import Row

__all__ = [
    "CAUSES",
    "CAUSE_NOTES",
    "classify_row",
    "detail",
    "pair_labels",
    "summary",
]

CAUSES: Final[tuple[str, ...]] = (
    "no_signal",
    "ambiguous",
    "weak_geometry",
    "hit",
)

CAUSE_NOTES: Final[dict[str, str]] = {
    "no_signal": "fewer than four matches; too little usable signal",
    "ambiguous": "matches found but the box lands off the instance",
    "weak_geometry": "instance found, box geometry too loose",
    "hit": "IoU >= 0.5",
}

ALIGNED_IOU: Final[float] = 0.1
NEAR_MISS_IOU: Final[float] = 0.3


def classify_row(row: Row) -> str:
    """Return the cause of one model's outcome on one pair.

    Args:
        row: Enriched per-sample row.

    Returns:
        One of :data:`CAUSES`.
    """
    if row.hit:
        return "hit"
    if row.matches < 4 or row.stage in {"no_matches", "too_few_matches"}:
        return "no_signal"
    if row.stage in {"ransac_failed"}:
        return "ambiguous"
    if row.stage in {"degenerate_homography", "no_prediction"}:
        return "weak_geometry"
    if not row.predicted:
        return "no_signal"
    return "ambiguous" if row.iou < ALIGNED_IOU else "weak_geometry"


def detail(row: Row) -> str:
    """Return a finer label within the coarse cause.

    Args:
        row: Enriched per-sample row.

    Returns:
        ``off_target`` or ``no_homography`` for ambiguous rows; ``loose`` or
        ``near_miss`` for weak geometry; empty otherwise.
    """
    cause = classify_row(row)
    if cause == "ambiguous":
        return "no_homography" if row.stage == "ransac_failed" else "off_target"
    if cause == "weak_geometry":
        return "loose" if row.iou < NEAR_MISS_IOU else "near_miss"
    return ""


def pair_labels(
    rows: Sequence[Row],
    models: Sequence[str],
) -> list[dict[str, object]]:
    """Label every positive pair, using its closest attempt across models.

    A pair is labelled from the model that came closest (highest IoU), since
    that model is the one that got furthest toward a solution and its failure
    mode is the informative one.

    Args:
        rows: Enriched rows.
        models: Model names.

    Returns:
        One record per positive pair.
    """
    grouped: dict[tuple[str, str], dict[str, Row]] = defaultdict(dict)
    for row in rows:
        if row.negative:
            continue
        grouped[(row.class_name, row.query)][row.model] = row

    records: list[dict[str, object]] = []
    for (class_name, query), by_model in grouped.items():
        present = [by_model[name] for name in models if name in by_model]
        if not present:
            continue
        best = max(present, key=lambda row: (row.iou, row.inliers))
        labels = [classify_row(row) for row in present]
        records.append(
            {
                "class_name": class_name,
                "query": query,
                "cause": classify_row(best),
                "detail": detail(best),
                "best_model": best.model,
                "best_iou": round(best.iou, 3),
                "matches": best.matches,
                "inliers": best.inliers,
                "coverage": round(best.coverage, 3),
                "stage": best.stage,
                "models_solved": sum(1 for label in labels if label == "hit"),
                "label_agreement": sum(
                    1 for label in labels if label == classify_row(best)
                ),
            }
        )
    return sorted(records, key=lambda record: (record["cause"], record["class_name"]))


def summary(
    records: Sequence[Mapping[str, object]],
    *,
    hard_only: bool = True,
) -> dict[str, object]:
    """Count causes over positive pairs or over pairs no model solves.

    Args:
        records: Records from :func:`pair_labels`.
        hard_only: Restrict to pairs with no successful model.

    Returns:
        Counts per cause, per class, and the selected total.
    """
    selected = [
        record
        for record in records
        if not hard_only or int(record["models_solved"]) == 0
    ]
    causes = Counter(str(record["cause"]) for record in selected)
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for record in selected:
        per_class[str(record["class_name"])][str(record["cause"])] += 1
    return {
        "total": len(selected),
        "causes": {cause: causes.get(cause, 0) for cause in CAUSES},
        "per_class": {name: dict(counts) for name, counts in per_class.items()},
    }
