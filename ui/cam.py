"""Grad-CAM saliency for the learned matchers.

The question behind the heatmap is which pixels the matcher's decision depends
on. Occlusion answers it by hiding regions and re-running; Grad-CAM answers it
from the gradients of the match score with respect to an internal feature map,
in one forward and one backward pass, and can be pointed at any layer.

Only the LightGlue extractors expose a usable feature map through this path.
Dense matchers (LoFTR, RoMa, FlexGlue) and the classical detectors return
``None``, and the UI keeps the occlusion sweep for those.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from benchmark.matchers.base import FeatureMatcher

__all__ = ["GRID", "gradcam", "list_layers"]

GRID = 64
LAYER_TYPES = (torch.nn.Conv2d, torch.nn.BatchNorm2d)


def list_layers(matcher: FeatureMatcher) -> list[dict[str, str]]:
    """Return the modules that can carry a Grad-CAM hook.

    Args:
        matcher: Matcher instance from the model registry.

    Returns:
        Layer records with ``name`` and ``kind``, in registration order.
    """
    extractor = getattr(matcher, "_extractor", None)
    if extractor is None:
        return []
    layers: list[dict[str, str]] = []
    for name, module in extractor.named_modules():
        if not name:
            continue
        if isinstance(module, LAYER_TYPES):
            layers.append({"name": name, "kind": type(module).__name__})
        elif _is_block(module):
            layers.append({"name": name, "kind": type(module).__name__})
    return layers


def _is_block(module: torch.nn.Module) -> bool:
    """Report whether a module is a container worth targeting.

    Args:
        module: Candidate module.

    Returns:
        True for modules that hold at least two submodules and are not leaves.
    """
    children = list(module.children())
    return len(children) >= 2 and not isinstance(module, torch.nn.Sequential)


def gradcam(
    matcher: FeatureMatcher,
    reference: np.ndarray,
    query: np.ndarray,
    layer: str | None = None,
) -> dict[str, object] | None:
    """Compute Grad-CAM maps for a reference/query pair.

    Args:
        matcher: Matcher instance from the model registry.
        reference: RGB reference crop.
        query: RGB query image.
        layer: Module name from :func:`list_layers`; the deepest convolution is
            used when omitted.

    Returns:
        ``{"reference": grid, "query": grid, "layer": name}`` with
        ``GRID x GRID`` normalized maps, or ``None`` when unsupported.
    """
    extractor = getattr(matcher, "_extractor", None)
    if extractor is None or not hasattr(matcher, "_matcher"):
        return None
    target = _resolve_layer(extractor, layer)
    if target is None:
        return None

    activations: list[torch.Tensor] = []

    def capture(_module: object, _inputs: object, output: object) -> None:
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.retain_grad()
            activations.append(output)

    handle = target.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            image0 = matcher._to_tensor(reference)
            image1 = matcher._to_tensor(query)
            features0 = _extract(extractor, image0)
            features1 = _extract(extractor, image1)
            # The vendored LightGlue detaches descriptors before matching, so a
            # matcher-score objective has no path back to the extractor. The
            # detector scores and descriptor norms do, and together they cover
            # both heads of the network.
            loss = _objective(features0, features1)
            loss.backward()
    except RuntimeError:
        return None
    finally:
        handle.remove()

    if len(activations) < 2:
        return None
    maps: list[list[list[float]]] = []
    for activation, image in zip(activations[:2], (reference, query)):
        gradient = activation.grad
        if gradient is None:
            return None
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0]
        raw = cam.detach().cpu().numpy().astype(np.float32)
        if raw.max() <= 0:
            raw = np.abs(raw)
        height, width = image.shape[:2]
        resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        sigma = max(2.0, 0.01 * max(width, height))
        maps.append(_grid(cv2.GaussianBlur(resized, (0, 0), sigma)))
    name = layer or _name_of(extractor, target)
    return {"reference": maps[0], "query": maps[1], "layer": name}


def _extract(extractor: torch.nn.Module, image: torch.Tensor) -> dict:
    """Run the extractor with gradients enabled.

    The vendored ``extract`` is wrapped in ``torch.no_grad()``, which kills the
    backward pass Grad-CAM needs. Its wrapped counterpart applies the same
    preprocessing and forward without that context.

    Args:
        extractor: Extractor module.
        image: Batched image tensor.

    Returns:
        Feature mapping as returned by the extractor.
    """
    extract = getattr(extractor, "extract")
    unwrapped = getattr(extract, "__wrapped__", None)
    if unwrapped is not None:
        return unwrapped(extractor, image)
    return extract(image)


def _objective(features0: dict, features1: dict) -> torch.Tensor:
    """Build the scalar Grad-CAM objective from extractor features.

    Args:
        features0: Reference features.
        features1: Query features.

    Returns:
        Sum of detector scores plus a small descriptor-norm term, so layers in
        both the score head and the descriptor head receive gradients.
    """
    total: torch.Tensor | None = None
    for features in (features0, features1):
        scores = features.get("keypoint_scores")
        if isinstance(scores, torch.Tensor):
            term = scores.sum()
            total = term if total is None else total + term
        descriptors = features.get("descriptors")
        if isinstance(descriptors, torch.Tensor) and descriptors.requires_grad:
            norm = descriptors.norm(dim=1).sum() * 0.1
            total = norm if total is None else total + norm
    if total is None:
        raise RuntimeError("extractor returned no differentiable features")
    return total


def _score_sum(output: object) -> torch.Tensor:
    """Reduce a matcher output to one scalar for backpropagation.

    Args:
        output: Matcher output mapping with per-batch score or match tensors.

    Returns:
        Sum of the match scores, or of the match indices when absent.
    """
    assert isinstance(output, dict)
    scores = output.get("scores")
    parts = scores if isinstance(scores, (list, tuple)) else [scores]
    total: torch.Tensor | None = None
    for part in parts:
        if part is None or not isinstance(part, torch.Tensor):
            continue
        total = part.sum() if total is None else total + part.sum()
    if total is not None:
        return total
    matches = output.get("matches")
    pieces = matches if isinstance(matches, (list, tuple)) else [matches]
    for piece in pieces:
        if isinstance(piece, torch.Tensor):
            total = (
                piece.float().sum() if total is None else total + piece.float().sum()
            )
    if total is None:
        raise RuntimeError("matcher output has no scores")
    return total


def _resolve_layer(
    extractor: torch.nn.Module,
    layer: str | None,
) -> torch.nn.Module | None:
    """Return the module to hook.

    Args:
        extractor: Extractor module.
        layer: Requested module name, or ``None`` for the deepest convolution.

    Returns:
        The module, or ``None`` when it cannot be resolved.
    """
    if layer:
        for name, module in extractor.named_modules():
            if name == layer:
                return module
        return None
    found: torch.nn.Conv2d | None = None
    for candidate in extractor.modules():
        if isinstance(candidate, torch.nn.Conv2d):
            found = candidate
    return found


def _name_of(extractor: torch.nn.Module, target: torch.nn.Module) -> str:
    """Return the registered name of a module.

    Args:
        extractor: Extractor module.
        target: Module to name.

    Returns:
        Registered name, or an empty string.
    """
    for name, module in extractor.named_modules():
        if module is target:
            return name
    return ""


def _grid(cam: np.ndarray, size: int = GRID) -> list[list[float]]:
    """Downsample a map to a normalized square grid.

    Args:
        cam: Non-negative saliency map.
        size: Grid side.

    Returns:
        ``size x size`` values in ``[0, 1]``.
    """
    if cam.size == 0 or float(cam.max()) <= 0:
        return [[0.0] * size for _ in range(size)]
    low = float(np.percentile(cam, 5))
    high = float(np.percentile(cam, 99.5))
    span = high - low
    if span <= 1e-9:
        normalized = np.zeros_like(cam, dtype=np.float32)
    else:
        normalized = np.clip((cam - low) / span, 0.0, 1.0).astype(np.float32)
    resized = cv2.resize(normalized, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0.0, 1.0).round(3).tolist()
