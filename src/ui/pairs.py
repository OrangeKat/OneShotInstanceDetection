"""Pair listing for the inspector UI."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.utils.dataset import InstanceDataset, InstanceSample

__all__ = ["PairEntry", "hard_pair_keys", "list_pairs"]


@dataclass(frozen=True)
class PairEntry:
    """One manifest pair as presented in the UI.

    Attributes:
        index: Position in the manifest.
        instance_id: Instance identifier.
        query: Query file name.
        negative: Whether the query has no ground-truth boxes.
        hard: Whether no benchmarked model solves this pair.
        gt_count: Number of ground-truth boxes.
    """

    index: int
    instance_id: str
    query: str
    negative: bool
    hard: bool
    gt_count: int


def hard_pair_keys(csv_paths: Iterable[Path]) -> set[tuple[str, str]]:
    """Read ``hard_pairs.csv`` files into ``(instance, query)`` keys.

    Args:
        csv_paths: Paths to hard-pair CSVs; missing files are skipped.

    Returns:
        Keys naming the pairs that no model solves.
    """
    keys: set[tuple[str, str]] = set()
    for path in csv_paths:
        if not Path(path).exists():
            continue
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                instance = row.get("class_name") or row.get("instance_id")
                query = row.get("query")
                if instance and query:
                    keys.add((str(instance), str(query)))
    return keys


def list_pairs(
    dataset: InstanceDataset,
    hard_keys: set[tuple[str, str]],
) -> list[PairEntry]:
    """Build the pair list shown in the sidebar.

    Args:
        dataset: Loaded manifest metadata.
        hard_keys: Pairs that no model solves.

    Returns:
        One entry per manifest pair, in manifest order.
    """
    entries: list[PairEntry] = []
    for index, sample in enumerate(dataset.samples):
        entries.append(
            PairEntry(
                index=index,
                instance_id=sample.instance_id,
                query=sample.query_path.name,
                negative=not sample.query_boxes,
                hard=(sample.instance_id, sample.query_path.name) in hard_keys,
                gt_count=len(sample.query_boxes),
            )
        )
    return entries


def filter_pairs(
    entries: Sequence[PairEntry],
    *,
    search: str = "",
    hard_only: bool = False,
    positives_only: bool = False,
    limit: int | None = None,
) -> list[PairEntry]:
    """Filter the pair list for the sidebar.

    Args:
        entries: All pairs.
        search: Case-insensitive substring of the instance or query name.
        hard_only: Keep only pairs no model solves.
        positives_only: Drop queries without ground truth.
        limit: Optional cap on the returned count.

    Returns:
        The filtered pairs.
    """
    needle = search.strip().lower()
    selected = [
        entry
        for entry in entries
        if (not hard_only or entry.hard)
        and (not positives_only or not entry.negative)
        and (
            not needle
            or needle in entry.instance_id.lower()
            or needle in entry.query.lower()
        )
    ]
    return selected[:limit] if limit is not None else selected


def sample_at(dataset: InstanceDataset, index: int) -> InstanceSample:
    """Return a sample by manifest index.

    Args:
        dataset: Loaded manifest metadata.
        index: Manifest position.

    Returns:
        The sample, without images loaded.

    Raises:
        IndexError: If the index is outside the manifest.
    """
    if index < 0 or index >= len(dataset.samples):
        raise IndexError(f"pair index out of range: {index}")
    return dataset.samples[index]
