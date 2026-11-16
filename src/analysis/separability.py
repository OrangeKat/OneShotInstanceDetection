"""Score separability: how far a matcher signal could drive a verifier.

The benchmark asks one question of every model: did the box land on the
instance? A verifier asks a different one: is this correspondence set real? The
gap between the two is measurable without training anything, by asking how well
each recorded score separates correct localizations from incorrect ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from src.analysis.data import Row

_SCORES: dict[str, Callable[[Row], float]] = {
    "inliers": lambda row: float(row.inliers),
    "matches": lambda row: float(row.matches),
    "inlier_ratio": lambda row: float(row.inlier_ratio),
    "coverage": lambda row: float(row.coverage),
    "inverse_reprojection_error": lambda row: (
        -float(row.reprojection_error) if row.reprojection_error is not None else -1e9
    ),
}

TARGETS = ("positive_vs_negative", "hit_vs_miss_positives")


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Return the rank-based ROC AUC with average ranks for ties.

    Args:
        scores: Higher means more likely to be positive.
        labels: Boolean truth.

    Returns:
        AUC in ``[0, 1]``, or ``nan`` when one class is missing.
    """
    values = np.asarray(scores, dtype=float)
    truth = np.asarray(labels, dtype=bool)
    positive = values[truth]
    negative = values[~truth]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = _average_ranks(combined)
    rank_sum = float(ranks[: positive.size].sum())
    return (rank_sum - positive.size * (positive.size + 1) / 2) / (
        positive.size * negative.size
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks, averaging tied values.

    Args:
        values: Numeric values.

    Returns:
        Rank array aligned with ``values``.
    """
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def separability_rows(
    rows: Sequence[Row],
    models: Sequence[str],
) -> list[dict[str, object]]:
    """Score how well each matcher signal separates two label definitions.

    Args:
        rows: Enriched rows.
        models: Model names.

    Returns:
        One record per model, target, and score.
    """
    records: list[dict[str, object]] = []
    for model in models:
        model_rows = [row for row in rows if row.model == model]
        positives = [row for row in model_rows if not row.negative]
        for target in TARGETS:
            subset = model_rows if target == "positive_vs_negative" else positives
            labels = [
                (not row.negative)
                if target == "positive_vs_negative"
                else bool(row.hit)
                for row in subset
            ]
            n_positive = sum(labels)
            n_negative = len(labels) - n_positive
            for name, score in _SCORES.items():
                records.append(
                    {
                        "model": model,
                        "target": target,
                        "score": name,
                        "auc": roc_auc([score(row) for row in subset], labels),
                        "positives": n_positive,
                        "negatives": n_negative,
                    }
                )
    return records


def best_separability(
    records: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Return the strongest single score per model and target.

    Args:
        records: Records from :func:`separability_rows`.

    Returns:
        Mapping of model to target to the best-scoring record.
    """
    best: dict[str, dict[str, object]] = {}
    for record in records:
        auc = float(record["auc"])
        if not np.isfinite(auc):
            continue
        model = str(record["model"])
        target = str(record["target"])
        current = best.get(model, {}).get(target)
        if current is None or auc > float(current["auc"]):
            best.setdefault(model, {})[target] = record
    return best
