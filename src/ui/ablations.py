"""Ablation runs: discovery, plan expansion, and execution."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ABLATION_FACTORS",
    "AblationJob",
    "JobQueue",
    "RunRecord",
    "discover_runs",
    "expand_plan",
    "flatten_run",
    "format_value",
]

ABLATION_FACTORS: dict[str, str] = {
    "max_keypoints": "--max-keypoints",
    "ransac_threshold": "--ransac-threshold",
    "max_side": "--max-side",
    "ratio": "--ratio",
}
DEFAULTS = {
    "max_keypoints": 2048,
    "ransac_threshold": 3.0,
    "max_side": 640,
    "ratio": 0.75,
}


@dataclass
class RunRecord:
    """One completed benchmark run on disk.

    Attributes:
        run_id: Output directory name.
        path: Directory holding ``results.json``.
        dataset: Manifest file name the run used.
        models: Models present in the run.
        config: Recorded run configuration.
        metrics: Model to summary metric mapping.
        modified: Last modification time of ``results.json``.
    """

    run_id: str
    path: Path
    dataset: str
    models: tuple[str, ...]
    config: Mapping[str, object]
    metrics: Mapping[str, Mapping[str, object]]
    modified: float


def discover_runs(root: Path) -> list[RunRecord]:
    """Find benchmark runs under a directory tree.

    Args:
        root: Directory searched for ``results.json`` files.

    Returns:
        Runs sorted by modification time, newest first.
    """
    records: list[RunRecord] = []
    for results_path in sorted(root.glob("**/results.json")):
        try:
            payload = json.loads(results_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = payload.get("config", {})
        metrics: dict[str, dict[str, object]] = {}
        for report in payload.get("models", []):
            latency = report.get("latency_ms") or {}
            metrics[str(report["model"])] = {
                "hit_rate": report.get("hit_rate"),
                "ap50": report.get("ap50"),
                "mean_iou": report.get("mean_iou"),
                "p50": latency.get("p50") if isinstance(latency, dict) else None,
                "samples": report.get("samples"),
            }
        if not metrics:
            continue
        records.append(
            RunRecord(
                run_id=results_path.parent.name,
                path=results_path.parent,
                dataset=_dataset_name(str(config.get("manifest", ""))),
                models=tuple(metrics),
                config=config,
                metrics=metrics,
                modified=results_path.stat().st_mtime,
            )
        )
    return sorted(records, key=lambda record: record.modified, reverse=True)


def flatten_run(record: RunRecord) -> dict[str, object]:
    """Convert a run into a JSON-friendly row set.

    Args:
        record: Run on disk.

    Returns:
        One dictionary holding config, models, and per-model metrics.
    """
    config = {
        "max_keypoints": record.config.get("max_keypoints"),
        "ransac_threshold": record.config.get("ransac_threshold"),
        "max_side": record.config.get("max_side"),
        "ratio": record.config.get("ratio"),
        "samples": record.config.get("samples"),
        "seed": record.config.get("seed"),
    }
    return {
        "run_id": record.run_id,
        "dataset": record.dataset,
        "models": list(record.models),
        "config": config,
        "metrics": {model: dict(values) for model, values in record.metrics.items()},
        "modified": record.modified,
    }


@dataclass
class AblationJob:
    """A queued or finished ablation run.

    Attributes:
        job_id: Unique identifier.
        label: Human-readable description of the variant.
        config: Benchmark arguments for this job.
        output: Directory the run writes to.
        status: ``queued``, ``running``, ``done``, or ``failed``.
        log: Captured stdout and stderr.
        returncode: Process exit code once finished.
        created: Queue time.
    """

    job_id: str
    label: str
    config: Mapping[str, object]
    output: Path
    status: str = "queued"
    log: list[str] = field(default_factory=list)
    returncode: int | None = None
    created: float = field(default_factory=time.time)

    def row(self) -> dict[str, object]:
        """Return a JSON-friendly job summary.

        Returns:
            Job state, config, and the tail of its log.
        """
        return {
            "job_id": self.job_id,
            "label": self.label,
            "config": dict(self.config),
            "output": str(self.output),
            "status": self.status,
            "returncode": self.returncode,
            "log": "".join(self.log[-40:]),
        }


class JobQueue:
    """Run ablation jobs one at a time on the local GPU."""

    def __init__(self, repo_root: Path, manifest: Path) -> None:
        """Initialize the queue.

        Args:
            repo_root: Repository root, used as the working directory.
            manifest: Manifest every job runs against.
        """
        self.repo_root = repo_root
        self.manifest = manifest
        self.jobs: dict[str, AblationJob] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, job: AblationJob) -> AblationJob:
        """Queue a job.

        Args:
            job: Job to run.

        Returns:
            The queued job.
        """
        with self._lock:
            self.jobs[job.job_id] = job
            self._order.append(job.job_id)
        return job

    def snapshot(self) -> list[dict[str, object]]:
        """Return every job in submission order.

        Returns:
            Job summaries, newest last.
        """
        with self._lock:
            return [self.jobs[job_id].row() for job_id in self._order]

    def running(self) -> bool:
        """Report whether a job is currently executing.

        Returns:
            True while a run is in progress.
        """
        with self._lock:
            return any(job.status == "running" for job in self.jobs.values())

    def _loop(self) -> None:
        """Execute queued jobs serially."""
        while True:
            job = self._next()
            if job is None:
                time.sleep(0.4)
                continue
            self._execute(job)

    def _next(self) -> AblationJob | None:
        """Pop the oldest queued job.

        Returns:
            The next job, or ``None``.
        """
        with self._lock:
            for job_id in self._order:
                job = self.jobs[job_id]
                if job.status == "queued":
                    job.status = "running"
                    return job
        return None

    def _execute(self, job: AblationJob) -> None:
        """Run one job end to end.

        Args:
            job: Job to execute.
        """
        command = build_command(self.manifest, job.config, job.output)
        job.output.mkdir(parents=True, exist_ok=True)
        (job.output / "job.json").write_text(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "label": job.label,
                    "config": dict(job.config),
                    "command": command,
                    "created": job.created,
                    "git": _git_sha(self.repo_root),
                },
                indent=2,
            )
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                job.log.append(line)
            job.returncode = process.wait()
            job.status = "done" if job.returncode == 0 else "failed"
        except Exception as error:  # noqa: BLE001 - surface the failure in the UI
            job.log.append(f"launch failed: {error}\n")
            job.status = "failed"


def build_command(
    manifest: Path,
    config: Mapping[str, object],
    output: Path,
) -> list[str]:
    """Build the benchmark command for one ablation variant.

    Args:
        manifest: Manifest path.
        config: Factor values and run options.
        output: Output directory.

    Returns:
        Argument vector for ``main.py``.
    """
    command = [
        "uv",
        "run",
        "python",
        "main.py",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    models = config.get("models") or ["sift"]
    command += ["--models", *[str(model) for model in models]]
    flag_map = {
        "max_keypoints": "--max-keypoints",
        "ransac_threshold": "--ransac-threshold",
        "max_side": "--max-side",
        "ratio": "--ratio",
    }
    for key, flag in flag_map.items():
        value = config.get(key)
        if value is not None:
            command += [flag, format_value(value)]
    if config.get("max_samples"):
        command += ["--max-samples", format_value(config["max_samples"])]
    return command


def format_value(value: object) -> str:
    """Format a factor value for a command line or a directory name.

    Integer-valued flags reject ``"256.0"``, so whole numbers are rendered
    without a decimal part.

    Args:
        value: Numeric factor value.

    Returns:
        Command-line string for the value.
    """
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def expand_plan(
    factor: str,
    values: Sequence[float | int],
    base: Mapping[str, object],
) -> list[dict[str, object]]:
    """Expand one factor into a list of run configurations.

    Args:
        factor: Factor name from :data:`ABLATION_FACTORS`.
        values: Values to test for that factor.
        base: Base configuration shared by every variant.

    Returns:
        One configuration per value, labelled by the factor.

    Raises:
        KeyError: If the factor is unknown.
    """
    if factor not in ABLATION_FACTORS:
        raise KeyError(f"unknown factor: {factor}")
    plan: list[dict[str, object]] = []
    for value in values:
        config = dict(base)
        config[factor] = value
        config["label"] = f"{factor}={value}"
        plan.append(config)
    return plan


def _dataset_name(manifest: str) -> str:
    """Return a readable dataset label for a manifest path.

    Args:
        manifest: Recorded manifest path.

    Returns:
        Parent directory name for a generic ``manifest.json``, otherwise the
        file name.
    """
    path = Path(manifest)
    if not manifest:
        return "unknown"
    if path.stem.startswith("manifest"):
        suffix = path.stem.removeprefix("manifest").lstrip("-")
        parent = path.parent.name
        return f"{parent}-{suffix}" if suffix else parent
    return path.name


def _git_sha(repo_root: Path) -> str:
    """Return the current commit, or an empty string outside git.

    Args:
        repo_root: Repository root.

    Returns:
        Short commit hash, or ``""``.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()
