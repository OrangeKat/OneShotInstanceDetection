"""Save and load the detector, locally or on the Hugging Face Hub.

Only the trained parts are stored: the decoder and the two heads. The backbone
is a published DINOv3 checkpoint, referenced by name, so the upload stays small
and the upstream weights keep their own licence.

Usage:
    from osid.model.hub import push_to_hub

    model = DetectorDescriptor()
    push_to_hub(model, "your-name/osid-boundary", private=True)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from osid.model.config import ModelConfig
from osid.model.detector import DetectorDescriptor

__all__ = ["from_pretrained", "load_local", "push_to_hub", "save_local"]

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.pt"


def save_local(model: DetectorDescriptor, directory: Path) -> Path:
    """Write the trained weights and the architecture config.

    Args:
        model: Model to save.
        directory: Destination directory.

    Returns:
        The directory written to.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_FILE).write_text(
        json.dumps(asdict(model.config), indent=2) + "\n"
    )
    state = {
        "decoder": model.decoder.state_dict(),
        "boundary": model.boundary.state_dict(),
        "descriptor": model.descriptor.state_dict(),
    }
    torch.save(state, directory / WEIGHTS_FILE)
    return directory


def load_local(directory: Path, *, device: str | None = None) -> DetectorDescriptor:
    """Load a model saved with :func:`save_local`.

    Args:
        directory: Directory holding ``config.json`` and ``model.pt``.
        device: Device override.

    Returns:
        The restored model.
    """
    directory = Path(directory)
    payload = json.loads((directory / CONFIG_FILE).read_text())
    config = ModelConfig(**payload)
    model = DetectorDescriptor(config, device=device)
    state = torch.load(directory / WEIGHTS_FILE, map_location=model.device)
    model.decoder.load_state_dict(state["decoder"])
    model.boundary.load_state_dict(state["boundary"])
    model.descriptor.load_state_dict(state["descriptor"])
    model.decoder.eval()
    model.boundary.eval()
    model.descriptor.eval()
    return model


def push_to_hub(
    model: DetectorDescriptor,
    repo_id: str,
    *,
    private: bool = True,
    directory: Path | None = None,
) -> str:
    """Upload the trained weights to a Hugging Face model repository.

    Args:
        model: Model to upload.
        repo_id: ``owner/name`` on the Hub.
        private: Create the repository as private.
        directory: Staging directory; a temporary one is used when omitted.

    Returns:
        The repository URL.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "pushing to the Hub needs `uv run --with huggingface_hub ...`"
        ) from error
    staging = save_local(model, directory or Path("outputs/osid-hub"))
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(staging),
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}"


def from_pretrained(repo_id: str, *, device: str | None = None) -> DetectorDescriptor:
    """Download a published model and load it.

    Args:
        repo_id: ``owner/name`` on the Hub.
        device: Device override.

    Returns:
        The restored model.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "loading from the Hub needs `uv run --with huggingface_hub ...`"
        ) from error
    local = snapshot_download(repo_id)
    return load_local(Path(local), device=device)
