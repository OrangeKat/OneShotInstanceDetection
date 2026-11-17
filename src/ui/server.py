"""Local HTTP server for the pair inspector and reference picker."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from src.models.feature_matching.base import FeatureMatcher
from src.pipeline import MODELS
from src.ui.ablations import (
    ABLATION_FACTORS,
    AblationJob,
    JobQueue,
    discover_runs,
    expand_plan,
    flatten_run,
    format_value,
)
from src.ui.diagnostics import build_dump, cache_path, oci_heatmap
from src.ui.pairs import PairEntry, filter_pairs, hard_pair_keys, list_pairs, sample_at
from src.ui.references import (
    SELECTION_PATH,
    candidates_for,
    crop_bytes,
    is_selectable,
    load_selection,
    save_selection,
    selection_for,
)
from src.utils.dataset import InstanceDataset

__all__ = ["InspectorState", "make_server", "serve"]

STATIC_DIR = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


@dataclass
class DatasetContext:
    """One manifest exposed by the inspector.

    Attributes:
        name: Short dataset key used in URLs.
        manifest: Manifest path.
        dataset: Parsed manifest.
        entries: Pair list for the sidebar.
        hard_keys: Pairs no model solves.
    """

    name: str
    manifest: Path
    dataset: InstanceDataset
    entries: list[PairEntry]
    hard_keys: set[tuple[str, str]]
    selectable: bool = False

    @property
    def data_dir(self) -> Path:
        """Return the dataset root directory.

        Returns:
            Directory holding the manifest and images.
        """
        return self.manifest.parent


class InspectorState:
    """Shared state for one inspector session.

    Attributes:
        contexts: Dataset key to context.
        models: Model names exposed in the UI.
        max_keypoints: Keypoint budget used for every run.
        ransac_threshold: RANSAC threshold in pixels.
        cache_dir: Directory for dumped pair payloads.
        runs_root: Directory scanned for completed benchmark runs.
    """

    def __init__(
        self,
        manifests: Mapping[str, Path],
        *,
        models: Sequence[str],
        hard_csv: Mapping[str, Sequence[Path]],
        cache_dir: Path,
        max_keypoints: int = 2048,
        ransac_threshold: float = 3.0,
        runs_root: Path | None = None,
        selection_path: Path = SELECTION_PATH,
    ) -> None:
        """Initialize the shared state.

        Args:
            manifests: Dataset key to manifest path.
            models: Model names to expose.
            hard_csv: Dataset key to hard-pair CSV paths.
            cache_dir: Directory for cached dumps.
            max_keypoints: Keypoint budget per image.
            ransac_threshold: RANSAC threshold in pixels.
            runs_root: Directory scanned for completed benchmark runs.
            selection_path: Reference selection JSON.
        """
        self.contexts: dict[str, DatasetContext] = {}
        for name, manifest in manifests.items():
            dataset = InstanceDataset(manifest, load_images=False)
            keys = hard_pair_keys(hard_csv.get(name, ()))
            self.contexts[name] = DatasetContext(
                name=name,
                manifest=manifest,
                dataset=dataset,
                entries=list_pairs(dataset, keys),
                hard_keys=keys,
                selectable=is_selectable(dataset, manifest.parent),
            )
        self.models = tuple(models)
        self.max_keypoints = max_keypoints
        self.ransac_threshold = ransac_threshold
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runs_root = runs_root or Path("outputs")
        self.repo_root = Path(__file__).resolve().parents[2]
        self.selection_path = selection_path
        self.selection = load_selection(selection_path)
        self._matchers: dict[str, FeatureMatcher] = {}
        self._dumps: dict[tuple[str, int, str], dict[str, object]] = {}
        self._lock = threading.Lock()
        default = next(iter(self.contexts))
        self.jobs = JobQueue(self.repo_root, self.contexts[default].manifest)

    # ------------------------------------------------------------- datasets
    def dataset_names(self) -> list[str]:
        """Return the available dataset keys.

        Returns:
            Dataset keys in configuration order.
        """
        return list(self.contexts)

    def context(self, name: str | None) -> DatasetContext:
        """Return a dataset context.

        Args:
            name: Dataset key, or ``None`` for the first one.

        Returns:
            The matching context.

        Raises:
            KeyError: If the dataset is unknown.
        """
        key = name or next(iter(self.contexts))
        if key not in self.contexts:
            raise KeyError(f"unknown dataset: {key}")
        return self.contexts[key]

    def datasets(self) -> list[dict[str, object]]:
        """Summarize every dataset for the client.

        Returns:
            One record per dataset.
        """
        return [
            {
                "name": context.name,
                "pairs": len(context.entries),
                "hard": len(context.hard_keys),
                "selectable": context.selectable,
                "manifest": str(context.manifest),
            }
            for context in self.contexts.values()
        ]

    # -------------------------------------------------------------- matcher
    def matcher(self, name: str) -> FeatureMatcher:
        """Return a cached matcher, building it on first use.

        Args:
            name: Model name.

        Returns:
            The matcher instance.

        Raises:
            KeyError: If the model is not exposed.
        """
        if name not in self.models:
            raise KeyError(f"unknown model: {name}")
        if name not in self._matchers:
            self._matchers[name] = MODELS[name](
                max_keypoints=self.max_keypoints,
                ransac_threshold=self.ransac_threshold,
            )
        return self._matchers[name]

    # --------------------------------------------------------------- dumps
    def dump(
        self,
        dataset_name: str | None,
        index: int,
        model: str,
        *,
        refresh: bool = False,
    ) -> dict[str, object]:
        """Return the diagnostic dump for a pair and model.

        Args:
            dataset_name: Dataset key.
            index: Manifest index.
            model: Model name.
            refresh: Recompute instead of using the cache.

        Returns:
            The dump payload.
        """
        context = self.context(dataset_name)
        key = (context.name, index, model)
        if not refresh and key in self._dumps:
            return self._dumps[key]
        sample = context.dataset.load(sample_at(context.dataset, index))
        with self._lock:
            result = self.matcher(model).match(
                context.dataset.reference_crop(sample),
                _query_image(sample),
            )
            payload = build_dump(
                context.dataset,
                sample,
                result,
                model=model,
                index=index,
            )
        payload["dataset"] = context.name
        self._dumps[key] = payload
        cache_path(self.cache_dir, model, index).write_text(json.dumps(payload))
        return payload

    def occlusion(
        self,
        dataset_name: str | None,
        index: int,
        model: str,
        grid: int,
    ) -> dict[str, object]:
        """Run the occlusion sweep for a pair and model.

        Args:
            dataset_name: Dataset key.
            index: Manifest index.
            model: Model name.
            grid: Cells per side.

        Returns:
            Heatmap payload.
        """
        context = self.context(dataset_name)
        sample = context.dataset.load(sample_at(context.dataset, index))
        with self._lock:
            matcher = self.matcher(model)
            baseline = matcher.match(
                context.dataset.reference_crop(sample),
                _query_image(sample),
            )
            return oci_heatmap(
                context.dataset,
                sample,
                matcher,
                baseline,
                grid=max(2, min(grid, 12)),
            )

    # ---------------------------------------------------------- references
    def references(self, dataset_name: str | None) -> dict[str, object]:
        """List instances with their current reference and stored choice.

        Args:
            dataset_name: Dataset key.

        Returns:
            Instances with selection state.
        """
        context = self.context(dataset_name)
        if not context.selectable:
            return {
                "dataset": context.name,
                "selectable": False,
                "instances": [],
                "statuses": list(_statuses()),
            }
        instances: dict[str, dict[str, object]] = {}
        for sample in context.dataset.samples:
            name = sample.instance_id
            if name in instances:
                continue
            record = selection_for(context.name, name, self.selection)
            instances[name] = {
                "instance": name,
                "image": sample.reference_path.name,
                "box": list(sample.reference_box) if sample.reference_box else None,
                "pairs": 0,
                "selected_image": (record or {}).get("image"),
                "status": (record or {}).get("status", "ok"),
                "note": (record or {}).get("note", ""),
            }
        for entry in context.entries:
            if entry.instance_id in instances:
                instances[entry.instance_id]["pairs"] += 1
        return {
            "dataset": context.name,
            "selectable": True,
            "instances": sorted(instances.values(), key=lambda item: item["instance"]),
            "statuses": list(_statuses()),
        }

    def reference_candidates(
        self,
        dataset_name: str | None,
        instance: str,
    ) -> dict[str, object]:
        """List candidate reference crops for one instance.

        Args:
            dataset_name: Dataset key.
            instance: Instance identifier.

        Returns:
            Candidates with display statistics.
        """
        context = self.context(dataset_name)
        candidates = candidates_for(context.dataset, instance, context.data_dir)
        return {
            "dataset": context.name,
            "instance": instance,
            "candidates": [
                {
                    "image": candidate.image,
                    "box": list(candidate.box) if candidate.box else None,
                    "width": int(candidate.crop_size[0]),
                    "height": int(candidate.crop_size[1]),
                    "current": candidate.is_current,
                    "url": _candidate_url(candidate),
                }
                for candidate in candidates
            ],
        }

    def choose_reference(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Store a reference choice or defect note.

        Args:
            payload: Request body with dataset, instance, and the record.

        Returns:
            The stored record.

        Raises:
            KeyError: If the dataset is unknown.
            ValueError: If the instance or status is invalid.
        """
        context = self.context(str(payload.get("dataset") or ""))
        instance = str(payload.get("instance") or "")
        if not any(
            sample.instance_id == instance for sample in context.dataset.samples
        ):
            raise ValueError(f"unknown instance: {instance}")
        status = str(payload.get("status") or "ok")
        if status not in _statuses():
            raise ValueError(f"unknown status: {status}")
        record: dict[str, object] = {
            "status": status,
            "note": str(payload.get("note") or ""),
        }
        image = payload.get("image")
        box = payload.get("box")
        if image:
            record["image"] = str(image)
        if isinstance(box, list) and len(box) == 4:
            record["box"] = [float(value) for value in box]
        self.selection.setdefault(context.name, {})[instance] = record
        save_selection(self.selection, self.selection_path)
        return {"dataset": context.name, "instance": instance, "record": record}

    # ---------------------------------------------------------- ablations
    def ablations(self) -> dict[str, object]:
        """Return completed runs, queued jobs, and the supported factors.

        Returns:
            Payload for the ablation view.
        """
        runs = [flatten_run(record) for record in discover_runs(self.runs_root)]
        return {
            "runs": runs,
            "jobs": self.jobs.snapshot(),
            "factors": sorted(ABLATION_FACTORS),
            "models": list(self.models),
        }

    def submit_ablation(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Expand one factor into runs and queue them.

        Args:
            payload: Request body with ``factor``, ``values``, and options.

        Returns:
            The queued job summaries.

        Raises:
            ValueError: If the request is malformed.
        """
        factor = str(payload.get("factor", ""))
        raw_values = payload.get("values")
        if isinstance(raw_values, str):
            raw_values = [part for part in raw_values.replace(",", " ").split() if part]
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("values must be a non-empty list")
        if factor not in ABLATION_FACTORS:
            raise ValueError(f"unknown factor: {factor}")
        models = payload.get("models") or ["sift"]
        if isinstance(models, str):
            models = [models]
        base: dict[str, object] = {
            "models": [str(model) for model in models],
            "max_samples": payload.get("max_samples"),
            "max_keypoints": payload.get("max_keypoints", self.max_keypoints),
            "ransac_threshold": payload.get("ransac_threshold", self.ransac_threshold),
        }
        plan = expand_plan(factor, [float(value) for value in raw_values], base)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        queued: list[dict[str, object]] = []
        for entry in plan:
            value = entry[factor]
            label = f"{factor}={format_value(value)}"
            output = (
                Path("outputs")
                / "ablations"
                / f"{stamp}-{factor}-{format_value(value)}"
            )
            job = AblationJob(
                job_id=f"{stamp}-{factor}-{format_value(value)}",
                label=label,
                config=entry,
                output=output,
            )
            queued.append(self.jobs.submit(job).row())
        return {"jobs": queued}


def _statuses() -> tuple[str, ...]:
    """Return the allowed reference statuses.

    Returns:
        Status names accepted by the API.
    """
    return ("ok", "low-res", "occluded", "wrong-content", "unusable")


def _candidate_url(candidate: object) -> str:
    """Build the image URL for a candidate crop.

    Args:
        candidate: Candidate object with image, box, and dataset fields.

    Returns:
        Relative URL serving the crop.
    """
    image = getattr(candidate, "image")
    box = getattr(candidate, "box")
    parts = [f"image={image}"]
    if box is not None:
        parts.append("box=" + ",".join(f"{float(value):.2f}" for value in box))
    return "/img/candidate?" + "&".join(parts)


def _load(sample: object) -> np.ndarray:
    """Load a sample's query image.

    Args:
        sample: Instance sample.

    Returns:
        RGB query image.
    """
    query_path = getattr(sample, "query_path")
    image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {query_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _query_image(sample: object) -> np.ndarray:
    """Return the query image of a sample, loading it if needed.

    Args:
        sample: Instance sample.

    Returns:
        RGB query image.
    """
    loaded = getattr(sample, "query_image", None)
    return loaded if loaded is not None else _load(sample)


def _encode(image: np.ndarray, *, qualities: int = 90) -> bytes:
    """Encode an RGB image as JPEG bytes.

    Args:
        image: RGB image.
        qualities: JPEG quality.

    Returns:
        Encoded bytes.
    """
    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), qualities],
    )
    if not ok:
        raise ValueError("image encoding failed")
    return buffer.tobytes()


class InspectorHandler(BaseHTTPRequestHandler):
    """Serve the inspector page and its JSON API."""

    state: InspectorState
    server_version = "InstanceInspector/2.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default per-request logging."""
        return

    def do_POST(self) -> None:  # noqa: N802
        """Route a POST request."""
        parsed = urlparse(self.path)
        try:
            payload = self._body()
            if parsed.path == "/api/ablation/run":
                self._send_json(self.state.submit_ablation(payload))
            elif parsed.path == "/api/references":
                self._send_json(self.state.choose_reference(payload))
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001 - surface errors to the UI
            self._send_json(
                {"error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_GET(self) -> None:  # noqa: N802
        """Route a GET request."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                self._send_file(STATIC_DIR / parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/models":
                self._send_json({"models": list(self.state.models)})
            elif parsed.path == "/api/datasets":
                self._send_json({"datasets": self.state.datasets()})
            elif parsed.path == "/api/pairs":
                self._pairs(query)
            elif parsed.path == "/api/pair":
                self._pair(query)
            elif parsed.path == "/api/occlusion":
                self._occlusion(query)
            elif parsed.path == "/api/ablations":
                self._send_json(self.state.ablations())
            elif parsed.path == "/api/references":
                self._send_json(self.state.references(_first(query, "dataset")))
            elif parsed.path == "/api/reference/candidates":
                self._send_json(
                    self.state.reference_candidates(
                        _first(query, "dataset"),
                        _first(query, "instance") or "",
                    )
                )
            elif parsed.path in {"/img/reference", "/img/query"}:
                self._image(parsed.path, query)
            elif parsed.path == "/img/candidate":
                self._candidate(query)
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except KeyError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except (IndexError, ValueError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as error:
            self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # noqa: BLE001 - surface errors to the UI
            self._send_json(
                {"error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _body(self) -> dict[str, object]:
        """Read a JSON request body.

        Returns:
            Parsed body, empty when absent.
        """
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _pairs(self, query: dict[str, list[str]]) -> None:
        """Return the filtered pair list.

        Args:
            query: Parsed query string.
        """
        context = self.state.context(_first(query, "dataset"))
        entries = filter_pairs(
            context.entries,
            search=_first(query, "search") or "",
            hard_only=_first(query, "hard") == "1",
            positives_only=_first(query, "positives", "1") != "0",
            limit=int(_first(query, "limit", "400") or 400),
        )
        self._send_json(
            {
                "dataset": context.name,
                "pairs": [
                    {
                        "index": entry.index,
                        "instance_id": entry.instance_id,
                        "query": entry.query,
                        "negative": entry.negative,
                        "hard": entry.hard,
                        "gt_count": entry.gt_count,
                    }
                    for entry in entries
                ],
                "total": len(context.entries),
                "hard_total": len(context.hard_keys),
            }
        )

    def _pair(self, query: dict[str, list[str]]) -> None:
        """Return the dump for one pair and model.

        Args:
            query: Parsed query string.
        """
        self._send_json(
            self.state.dump(
                _first(query, "dataset"),
                int(_first(query, "index", "0") or 0),
                _first(query, "model") or self.state.models[0],
                refresh=_first(query, "refresh") == "1",
            )
        )

    def _occlusion(self, query: dict[str, list[str]]) -> None:
        """Return the occlusion heatmap for one pair and model.

        Args:
            query: Parsed query string.
        """
        self._send_json(
            self.state.occlusion(
                _first(query, "dataset"),
                int(_first(query, "index", "0") or 0),
                _first(query, "model") or self.state.models[0],
                int(_first(query, "grid", "6") or 6),
            )
        )

    def _image(self, path: str, query: dict[str, list[str]]) -> None:
        """Return the reference crop or the query image.

        Args:
            path: Requested image route.
            query: Parsed query string.
        """
        context = self.state.context(_first(query, "dataset"))
        index = int(_first(query, "index", "0") or 0)
        sample = sample_at(context.dataset, index)
        if path == "/img/reference":
            image = context.dataset.reference_crop(context.dataset.load(sample))
        else:
            image = _query_image(sample)
        self._send_bytes(_encode(image), "image/jpeg")

    def _candidate(self, query: dict[str, list[str]]) -> None:
        """Return one candidate reference crop.

        Args:
            query: Parsed query string.
        """
        context = self.state.context(_first(query, "dataset"))
        image_name = _first(query, "image") or ""
        box_value = _first(query, "box")
        box = None
        if box_value:
            numbers = [float(value) for value in box_value.split(",")]
            if len(numbers) == 4:
                box = numbers
        from src.ui.references import Candidate  # local import: display helper

        path = _find_reference_image(context, image_name)
        width, height = 0, 0
        candidate = Candidate(
            image=image_name,
            box=box,
            path=path,
            width=width,
            height=height,
        )
        self._send_bytes(crop_bytes(candidate), "image/png")

    def _send_file(self, path: Path) -> None:
        """Send a static file.

        Args:
            path: File path inside the static directory.
        """
        resolved = path.resolve()
        if not resolved.is_file() or STATIC_DIR.resolve() not in resolved.parents:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(
            resolved.read_bytes(),
            CONTENT_TYPES.get(resolved.suffix, "application/octet-stream"),
        )

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a JSON response.

        Args:
            payload: JSON-serializable payload.
            status: HTTP status code.
        """
        self._send_bytes(
            json.dumps(payload).encode(),
            "application/json",
            status=status,
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        """Send raw bytes with headers.

        Args:
            body: Response body.
            content_type: MIME type.
            status: HTTP status code.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _find_reference_image(context: DatasetContext, image_name: str) -> Path:
    """Locate a class image inside a dataset.

    Args:
        context: Dataset context.
        image_name: Image file name.

    Returns:
        Path to the image.

    Raises:
        FileNotFoundError: If no matching image exists.
    """
    for sample in context.dataset.samples:
        if sample.reference_path.name == image_name:
            return sample.reference_path
    candidates = list(context.data_dir.glob(f"images/*/{image_name}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"image not found: {image_name}")


def _first(
    query: dict[str, list[str]], key: str, default: str | None = None
) -> str | None:
    """Return the first value of a query parameter.

    Args:
        query: Parsed query string.
        key: Parameter name.
        default: Fallback value.

    Returns:
        First value or the default.
    """
    values = query.get(key)
    return values[0] if values else default


def make_server(
    state: InspectorState,
    *,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    """Build a server bound to the inspector state.

    Args:
        state: Shared inspector state.
        host: Bind address.
        port: Bind port; ``0`` picks a free one.

    Returns:
        A configured, unstarted server.
    """
    handler = type("BoundInspectorHandler", (InspectorHandler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def serve(
    manifests: Mapping[str, Path],
    *,
    models: Sequence[str],
    hard_csv: Mapping[str, Sequence[Path]],
    cache_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_keypoints: int = 2048,
    runs_root: Path | None = None,
    selection_path: Path = SELECTION_PATH,
) -> int:
    """Run the inspector until interrupted.

    Args:
        manifests: Dataset key to manifest path.
        models: Model names to expose.
        hard_csv: Dataset key to hard-pair CSV paths.
        cache_dir: Directory for cached dumps.
        host: Bind address.
        port: Bind port.
        max_keypoints: Keypoint budget per image.
        runs_root: Directory scanned for completed runs.
        selection_path: Reference selection JSON.

    Returns:
        Process exit code.
    """
    state = InspectorState(
        manifests,
        models=models,
        hard_csv=hard_csv,
        cache_dir=cache_dir,
        max_keypoints=max_keypoints,
        runs_root=runs_root,
        selection_path=selection_path,
    )
    server = make_server(state, host=host, port=port)
    address, actual_port = server.server_address[:2]
    print(f"inspector: http://{address}:{actual_port}", flush=True)
    for context in state.contexts.values():
        print(
            f"  {context.name}: {len(context.entries)} pairs | "
            f"{len(context.hard_keys)} hard",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
