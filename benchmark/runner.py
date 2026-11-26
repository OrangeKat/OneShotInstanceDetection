"""Benchmark run loop over a manifest."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from benchmark.geometry import box_iou, predicted_box, query_image, reference_crop
from benchmark.matchers.base import FeatureMatcher
from benchmark.metrics import summarize
from benchmark.types import SampleResult
from utils.dataset import InstanceSample
from utils.diagnostics import match_diagnostics


def run_benchmark(
    matcher: FeatureMatcher,
    samples: Sequence[InstanceSample],
    iou_threshold: float,
    loader: Callable[[InstanceSample], InstanceSample] | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, object]:
    """Run one matcher over all samples.

    Args:
        matcher: Model instance.
        samples: Reference/query pairs.
        iou_threshold: IoU threshold that defines a hit.
        loader: Optional callable that loads a sample's images.
        progress: Optional callback called with the completed sample count.

    Returns:
        A model report.
    """
    records: list[SampleResult] = []
    for index, sample in enumerate(samples, start=1):
        if loader is not None:
            sample = loader(sample)
        reference = reference_crop(sample)
        query = query_image(sample)
        start = time.perf_counter()
        match = matcher.match(reference, query)
        latency_ms = (time.perf_counter() - start) * 1000.0
        box = predicted_box(match, reference.shape[:2])
        best_iou = (
            max(box_iou(box, gt) for gt in sample.query_boxes)
            if box is not None and sample.query_boxes
            else 0.0
        )
        diagnostics = match_diagnostics(
            match,
            reference.shape[:2],
            query.shape[:2],
            box,
            sample.query_boxes,
            best_iou,
            iou_threshold,
        )
        records.append(
            SampleResult(
                dataset=sample.dataset,
                instance_id=sample.instance_id,
                query=str(sample.query_path),
                latency_ms=latency_ms,
                match_count=match.match_count,
                inlier_count=match.inlier_count,
                best_iou=best_iou,
                predicted_box=box,
                gt_boxes=sample.query_boxes,
                hit=best_iou >= iou_threshold,
                **diagnostics,
            )
        )
        if progress is not None:
            progress(index)
    return summarize(matcher.name, records, iou_threshold)
