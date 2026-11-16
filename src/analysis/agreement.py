"""Cross-model agreement over reference/query pairs.

A pair that every matcher misses is evidence about the data, not about a model.
Keeping that set explicit is what separates "which model is best" from "what is
actually hard".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from src.analysis.data import Row


def pair_outcomes(
    rows: Sequence[Row],
    models: Sequence[str],
) -> list[dict[str, object]]:
    """Return per-pair hit counts across models.

    Only pairs evaluated by every model are kept, so the counts are comparable.

    Args:
        rows: Enriched rows.
        models: Model names.

    Returns:
        Pair records sorted by ascending hit count.
    """
    outcomes: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"hits": {}, "negative": True}
    )
    for row in rows:
        key = (row.class_name, row.query)
        entry = outcomes[key]
        hits = entry["hits"]
        if isinstance(hits, dict):
            hits[row.model] = int(row.hit)
        entry["negative"] = bool(entry["negative"]) and row.negative

    expected = len(models)
    records: list[dict[str, object]] = []
    for (class_name, query), entry in outcomes.items():
        hits = entry["hits"]
        if not isinstance(hits, dict) or len(hits) != expected:
            continue
        n_hit = sum(int(value) for value in hits.values())
        records.append(
            {
                "class_name": class_name,
                "query": query,
                "negative": bool(entry["negative"]),
                "models": expected,
                "models_hit": n_hit,
                "hit_fraction": n_hit / expected,
                "solved_by": " ".join(
                    sorted(model for model, value in hits.items() if int(value))
                ),
            }
        )
    return sorted(
        records,
        key=lambda record: (
            int(record["models_hit"]),
            str(record["class_name"]),
            str(record["query"]),
        ),
    )


def agreement_summary(
    pairs: Sequence[Mapping[str, object]],
    models: Sequence[str],
) -> dict[str, object]:
    """Summarize how many models solve each positive pair.

    Args:
        pairs: Records from :func:`pair_outcomes`.
        models: Model names.

    Returns:
        Summary with total pairs, oracle hit rate, hard-pair count, and the
        distribution of pairs by number of successful models.
    """
    positives = [pair for pair in pairs if not pair["negative"]]
    total = len(positives)
    distribution = [0] * (len(models) + 1)
    for pair in positives:
        distribution[int(pair["models_hit"])] += 1
    solved_by_all = distribution[-1] if distribution else 0
    solved_by_any = total - distribution[0] if distribution else 0
    return {
        "pairs": total,
        "negatives": len(pairs) - total,
        "solved_by_all": solved_by_all,
        "solved_by_any": solved_by_any,
        "hard_pairs": distribution[0] if distribution else 0,
        "oracle_hit_rate": solved_by_any / total if total else float("nan"),
        "distribution": distribution,
    }


def hard_pair_rows(
    pairs: Sequence[Mapping[str, object]],
    models: Sequence[str],
) -> list[dict[str, object]]:
    """Return positive pairs that no model solves.

    Args:
        pairs: Records from :func:`pair_outcomes`.
        models: Model names.

    Returns:
        Hard-pair records with class, query, and model count.
    """
    return [
        {
            "class_name": pair["class_name"],
            "query": pair["query"],
            "models": pair["models"],
            "models_hit": pair["models_hit"],
        }
        for pair in pairs
        if not pair["negative"] and int(pair["models_hit"]) == 0
    ]


def agreement_by_model(
    pairs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Count unique and shared solves per model.

    Args:
        pairs: Records from :func:`pair_outcomes`.

    Returns:
        One record per model with its solves, unique solves, and share of the
        pairs no other model solves.
    """
    positives = [pair for pair in pairs if not pair["negative"]]
    counts: dict[str, dict[str, int]] = {}
    for pair in positives:
        solved = str(pair["solved_by"]).split()
        for model in solved:
            entry = counts.setdefault(model, {"solved": 0, "unique": 0})
            entry["solved"] += 1
            if len(solved) == 1:
                entry["unique"] += 1
    return [
        {
            "model": model,
            "solved": values["solved"],
            "unique": values["unique"],
        }
        for model, values in sorted(counts.items())
    ]
