"""Local HTTP server for the pair inspector."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from src.models.feature_matching.base import FeatureMatcher
from src.pipeline import MODELS
from src.ui.diagnostics import build_dump, cache_path, oci_heatmap
from src.ui.pairs import filter_pairs, hard_pair_keys, list_pairs, sample_at
from src.utils.dataset import InstanceDataset

__all__ = ["InspectorState", "make_server", "serve"]

STATIC_DIR = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class InspectorState:
    """Shared state for one inspector session.

    Attributes:
        dataset: Manifest metadata.
        hard_keys: Pairs no model solves.
        models: Model names exposed in the UI.
        max_keypoints: Keypoint budget used for every run.
        ransac_threshold: RANSAC threshold in pixels.
        cache_dir: Directory for dumped pair payloads.
    """

    def __init__(
        self,
        manifest: Path,
        *,
        models: Sequence[str],
        hard_csv: Sequence[Path],
        cache_dir: Path,
        max_keypoints: int = 2048,
        ransac_threshold: float = 3.0,
    ) -> None:
        """Initialize the shared state.

        Args:
            manifest: Benchmark manifest.
            models: Model names to expose.
            hard_csv: Hard-pair CSVs produced by the analysis.
            cache_dir: Directory for cached dumps.
            max_keypoints: Keypoint budget per image.
            ransac_threshold: RANSAC threshold in pixels.
        """
        self.dataset = InstanceDataset(manifest, load_images=False)
        self.hard_keys = hard_pair_keys(hard_csv)
        self.entries = list_pairs(self.dataset, self.hard_keys)
        self.models = tuple(models)
        self.max_keypoints = max_keypoints
        self.ransac_threshold = ransac_threshold
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._matchers: dict[str, FeatureMatcher] = {}
        self._dumps: dict[tuple[int, str], dict[str, object]] = {}
        self._lock = threading.Lock()

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

    def dump(
        self, index: int, model: str, *, refresh: bool = False
    ) -> dict[str, object]:
        """Return the diagnostic dump for a pair and model.

        Args:
            index: Manifest index.
            model: Model name.
            refresh: Recompute instead of using the cache.

        Returns:
            The dump payload.
        """
        key = (index, model)
        if not refresh and key in self._dumps:
            return self._dumps[key]
        sample = self.dataset.load(sample_at(self.dataset, index))
        with self._lock:
            result = self.matcher(model).match(
                self.dataset.reference_crop(sample),
                _query_image(sample),
            )
            payload = build_dump(
                self.dataset,
                sample,
                result,
                model=model,
                index=index,
            )
        self._dumps[key] = payload
        cache_path(self.cache_dir, model, index).write_text(json.dumps(payload))
        return payload

    def occlusion(self, index: int, model: str, grid: int) -> dict[str, object]:
        """Run the occlusion sweep for a pair and model.

        Args:
            index: Manifest index.
            model: Model name.
            grid: Cells per side.

        Returns:
            Heatmap payload.
        """
        sample = self.dataset.load(sample_at(self.dataset, index))
        with self._lock:
            matcher = self.matcher(model)
            baseline = matcher.match(
                self.dataset.reference_crop(sample),
                _query_image(sample),
            )
            return oci_heatmap(
                self.dataset,
                sample,
                matcher,
                baseline,
                grid=max(2, min(grid, 12)),
            )


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
    server_version = "InstanceInspector/1.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default per-request logging."""
        return

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
            elif parsed.path == "/api/pairs":
                self._pairs(query)
            elif parsed.path == "/api/pair":
                self._pair(query)
            elif parsed.path == "/api/occlusion":
                self._occlusion(query)
            elif parsed.path in {"/img/reference", "/img/query"}:
                self._image(parsed.path, query)
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

    def _pairs(self, query: dict[str, list[str]]) -> None:
        """Return the filtered pair list.

        Args:
            query: Parsed query string.
        """
        entries = filter_pairs(
            self.state.entries,
            search=query.get("search", [""])[0],
            hard_only=query.get("hard", ["0"])[0] == "1",
            positives_only=query.get("positives", ["1"])[0] == "1",
            limit=int(query.get("limit", ["400"])[0]),
        )
        self._send_json(
            {
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
                "total": len(self.state.entries),
                "hard_total": len(self.state.hard_keys),
            }
        )

    def _pair(self, query: dict[str, list[str]]) -> None:
        """Return the dump for one pair and model.

        Args:
            query: Parsed query string.
        """
        index = int(query.get("index", ["0"])[0])
        model = query.get("model", [self.state.models[0]])[0]
        refresh = query.get("refresh", ["0"])[0] == "1"
        self._send_json(self.state.dump(index, model, refresh=refresh))

    def _occlusion(self, query: dict[str, list[str]]) -> None:
        """Return the occlusion heatmap for one pair and model.

        Args:
            query: Parsed query string.
        """
        index = int(query.get("index", ["0"])[0])
        model = query.get("model", [self.state.models[0]])[0]
        grid = int(query.get("grid", ["6"])[0])
        self._send_json(self.state.occlusion(index, model, grid))

    def _image(self, path: str, query: dict[str, list[str]]) -> None:
        """Return the reference crop or the query image.

        Args:
            path: Requested image route.
            query: Parsed query string.
        """
        index = int(query.get("index", ["0"])[0])
        sample = sample_at(self.state.dataset, index)
        if path == "/img/reference":
            image = self.state.dataset.reference_crop(self.state.dataset.load(sample))
        else:
            image = _query_image(sample)
        self._send_bytes(_encode(image), "image/jpeg")

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
    manifest: Path,
    *,
    models: Sequence[str],
    hard_csv: Sequence[Path],
    cache_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_keypoints: int = 2048,
) -> int:
    """Run the inspector until interrupted.

    Args:
        manifest: Benchmark manifest.
        models: Model names to expose.
        hard_csv: Hard-pair CSVs produced by the analysis.
        cache_dir: Directory for cached dumps.
        host: Bind address.
        port: Bind port.
        max_keypoints: Keypoint budget per image.

    Returns:
        Process exit code.
    """
    state = InspectorState(
        manifest,
        models=models,
        hard_csv=hard_csv,
        cache_dir=cache_dir,
        max_keypoints=max_keypoints,
    )
    server = make_server(state, host=host, port=port)
    address, actual_port = server.server_address[:2]
    print(f"inspector: http://{address}:{actual_port}", flush=True)
    print(
        f"pairs: {len(state.entries)} | hard: {len(state.hard_keys)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
