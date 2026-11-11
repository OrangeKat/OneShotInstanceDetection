"""Retrieval evaluation for the Paris6k protocol."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.models.feature_matching.base import FeatureMatcher

Box = tuple[float, float, float, float]
THRESHOLDS = (1, 5, 10)


@dataclass
class RetrievalQuery:
    """One query ROI and its candidate database images.

    Attributes:
        query_id: Query index in the manifest.
        query_path: Query image path.
        query_box: ROI in the query image.
        positives: Database images of the same landmark.
        candidates: Database images to rank (positives and negatives).
        metadata: Free-form metadata.
    """

    query_id: int
    query_path: Path
    query_box: Box
    positives: tuple[Path, ...]
    candidates: tuple[Path, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


def load_queries(
    manifest: str | Path,
    *,
    root: str | Path | None = None,
    limit: int | None = None,
) -> list[RetrievalQuery]:
    """Load a retrieval manifest.

    Args:
        manifest: JSON manifest path.
        root: Base directory for relative paths.
        limit: Optional query limit.

    Returns:
        Parsed queries.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If a record is malformed.
    """
    path = Path(manifest)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    base = Path(root) if root is not None else path.parent
    payload = json.loads(path.read_text())
    records = payload.get("queries", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("retrieval manifest must contain a list of queries")

    queries: list[RetrievalQuery] = []
    for index, record in enumerate(records):
        if limit is not None and len(queries) >= limit:
            break
        if not isinstance(record, Mapping):
            raise ValueError(f"query {index} is not an object")
        missing = {"query", "query_box", "positives", "candidates"} - set(record)
        if missing:
            raise ValueError(f"query {index} is missing keys: {sorted(missing)}")
        query_box = _parse_box(record["query_box"])
        positives = tuple(_resolve(value, base) for value in record["positives"])
        candidates = tuple(_resolve(value, base) for value in record["candidates"])
        queries.append(
            RetrievalQuery(
                query_id=int(record.get("query_id", index)),
                query_path=_resolve(record["query"], base),
                query_box=query_box,
                positives=positives,
                candidates=candidates,
                metadata=record.get("metadata", {}),
            )
        )
    return queries


def evaluate_model(
    matcher: FeatureMatcher,
    queries: Sequence[RetrievalQuery],
    thresholds: Sequence[int] = THRESHOLDS,
    progress: Callable[[int], None] | None = None,
) -> dict[str, object]:
    """Rank candidates for each query and compute retrieval metrics.

    Args:
        matcher: Feature matcher.
        queries: Retrieval queries.
        thresholds: Recall@k values.
        progress: Optional callback called with the completed query count.

    Returns:
        Per-model metrics and per-query records.
    """
    average_precisions: list[float] = []
    recalls: dict[int, list[bool]] = {k: [] for k in thresholds}
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    latencies: list[float] = []
    per_query: list[dict[str, object]] = []

    for index, query in enumerate(queries, start=1):
        image = _load_rgb(query.query_path)
        reference = _crop(image, query.query_box)
        positives = set(query.positives)
        scores: list[tuple[Path, float]] = []
        query_latencies: list[float] = []
        for candidate in dict.fromkeys(query.candidates):
            is_positive = candidate in positives
            candidate_image = _load_rgb(candidate)
            start = time.perf_counter()
            result = matcher.match(reference, candidate_image)
            latency = (time.perf_counter() - start) * 1000.0
            query_latencies.append(latency)
            latencies.append(latency)
            score = float(result.inlier_count) if result.homography is not None else 0.0
            scores.append((candidate, score))
            if is_positive:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

        scores.sort(key=lambda item: item[1], reverse=True)
        ranked = [candidate for candidate, _ in scores]
        relevant = [candidate in positives for candidate in ranked]
        ap = average_precision(relevant, len(positives))
        if ap is None:
            if progress is not None:
                progress(index)
            continue
        average_precisions.append(ap)
        query_recall: dict[str, bool] = {}
        for threshold in thresholds:
            hit = any(relevant[:threshold])
            recalls[threshold].append(hit)
            query_recall[str(threshold)] = hit
        per_query.append(
            {
                "query_id": query.query_id,
                "query": str(query.query_path),
                "ap": ap,
                "recall": query_recall,
                "top1": str(ranked[0]) if ranked else None,
                "top1_score": scores[0][1] if scores else 0.0,
                "candidates": len(scores),
                "positives": len(positives),
                "latency_ms": float(np.mean(query_latencies))
                if query_latencies
                else 0.0,
            }
        )
        if progress is not None:
            progress(index)

    return {
        "model": matcher.name,
        "queries": len(average_precisions),
        "candidates": sum(len(query.candidates) for query in queries),
        "map": _mean(average_precisions),
        "recall": {
            str(threshold): _mean(recalls[threshold]) for threshold in thresholds
        },
        "mean_inliers": {
            "positive": _mean(positive_scores),
            "negative": _mean(negative_scores),
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "per_query": per_query,
    }


def average_precision(relevance: Sequence[bool], positives: int) -> float | None:
    """Compute average precision from a ranked relevance list.

    Args:
        relevance: Relevance flags in rank order.
        positives: Number of relevant items in the candidate set.

    Returns:
        Average precision, or ``None`` when there are no positives.
    """
    if positives <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            hits += 1
            total += hits / rank
    return total / positives


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB.

    Args:
        path: Image path.

    Returns:
        RGB image.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Crop an image with clamping.

    Args:
        image: RGB image.
        box: ``[x1, y1, x2, y2]`` box.

    Returns:
        Cropped image.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(x1 + 1, min(width, round(x2))))
    y2 = int(max(y1 + 1, min(height, round(y2))))
    return image[y1:y2, x1:x2]


def _resolve(value: object, root: Path) -> Path:
    """Resolve a manifest path against a root.

    Args:
        value: Path value.
        root: Base directory.

    Returns:
        Absolute path.
    """
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _parse_box(value: object) -> Box:
    """Parse one ``[x1, y1, x2, y2]`` box.

    Args:
        value: Raw box value.

    Returns:
        Parsed box.

    Raises:
        ValueError: If the box is malformed.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"box must have 4 coordinates, got {value!r}")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid box: {(x1, y1, x2, y2)}")
    return (x1, y1, x2, y2)


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean or ``None`` for an empty sequence.

    Args:
        values: Numeric values.

    Returns:
        Mean value or ``None``.
    """
    return float(np.mean(values)) if values else None
