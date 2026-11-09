"""Markdown report for Paris6k retrieval evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

THRESHOLDS = (1, 5, 10)


def render_retrieval_report(
    results: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> str:
    """Render the retrieval report.

    Args:
        results: Per-model metric dictionaries.
        config: Run configuration.

    Returns:
        Markdown document.
    """
    recalls = " | ".join(f"R@{threshold}" for threshold in THRESHOLDS)
    lines = [
        "# Paris6k retrieval",
        "",
        f"- Data: `{config['manifest']}`",
        f"- Queries: `{config['queries']}`",
        f"- Candidates per query: "
        f"`{config['min_candidates']}`–`{config['max_candidates']}`",
        f"- Max keypoints: `{config['max_keypoints']}`",
        f"- RANSAC threshold: `{config['ransac_threshold']}` px",
        "",
        "The matcher ranks the candidate database images for each query ROI; "
        "the score is the RANSAC inlier count. Positives are the revisitop "
        "easy and hard images for that landmark.",
        "",
        "| Model | Queries | mAP | " + recalls + " | Mean inliers pos/neg "
        "| p50 ms | p95 ms |",
        "|---|---:|---:|" + "---:|" * len(THRESHOLDS) + "---:|---:|---:|",
    ]
    for result in results:
        recall = result["recall"]
        inliers = result["mean_inliers"]
        latency = result["latency_ms"]
        lines.append(
            f"| {result['model']} | {result['queries']} "
            f"| {_fmt(result['map'])} | "
            + " | ".join(_fmt(recall[str(k)]) for k in THRESHOLDS)
            + f" | {_fmt(inliers['positive'])} / {_fmt(inliers['negative'])} "
            f"| {latency['p50']:.1f} | {latency['p95']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            "- Each query is the ROI crop from a revisitop query image; "
            "candidates are the easy/hard positives plus sampled negatives.",
            "- mAP is computed over the candidate list, so it measures "
            "candidate reranking rather than full 6k retrieval.",
            "- Recall@k is the fraction of queries with at least one positive "
            "in the top k candidates.",
            "- Junk images are excluded from the candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    """Format an optional metric.

    Args:
        value: Metric value or ``None``.

    Returns:
        Formatted string.
    """
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
