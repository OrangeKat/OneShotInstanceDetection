"""Train the heads on synthetic shapes.

The backbone is frozen, so a step is two forward passes (one per view) through
ViT-S plus the two losses. That is small enough to run on one 16 GB card without
feature caching.

What is supervised:

- the boundary heatmap of both views, from the polygon silhouettes;
- descriptors at corresponding boundary points, from the known homography.

Usage:
    uv run python -m osid.model.train --steps 200
    uv run python -m osid.model.train --steps 50 --batch 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from osid.model.config import ModelConfig
from osid.model.detector import DetectorDescriptor
from osid.model.losses import boundary_target, focal_loss, infonce, symmetry_mask
from osid.model.matcher import extract_keypoints
from osid.model.synth import ShapePair, ShapePairGenerator

__all__ = ["Batch", "build_batch", "load_checkpoint", "main", "train"]


@dataclass
class Batch:
    """Tensors for one training pair.

    Attributes:
        image0: Reference view, ``(H, W, 3)`` uint8.
        image1: Query view.
        heat0: Boundary heatmap for view 0, ``(1, h, w)``.
        heat1: Boundary heatmap for view 1.
        points0: ``(K, 2)`` boundary points in view 0 pixels.
        points1: ``(K, 2)`` the same points in view 1 pixels.
        symmetry: ``(K, K)`` mask of ambiguous negatives.
    """

    image0: np.ndarray
    image1: np.ndarray
    heat0: np.ndarray
    heat1: np.ndarray
    points0: np.ndarray
    points1: np.ndarray
    symmetry: np.ndarray


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed polyline at evenly spaced arc-length positions.

    Args:
        points: ``(N, 2)`` vertices.
        count: Number of samples.

    Returns:
        ``(count, 2)`` points along the boundary.
    """
    closed = np.concatenate([points, points[:1]], axis=0)
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(segments.sum())
    if total <= 0:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    positions = np.linspace(0.0, total, count, endpoint=False)
    samples = []
    for position in positions:
        index = int(np.searchsorted(cumulative, position, side="right") - 1)
        index = min(index, len(points) - 1)
        local = (position - cumulative[index]) / max(segments[index], 1e-9)
        samples.append(closed[index] + local * (closed[index + 1] - closed[index]))
    return np.asarray(samples, dtype=np.float32)


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to 2-D points.

    Args:
        homography: ``3 x 3`` matrix.
        points: ``(N, 2)`` points.

    Returns:
        ``(N, 2)`` projected points.
    """
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ homography.T
    return projected[:, :2] / projected[:, 2:3]


def build_batch(
    pair: ShapePair,
    *,
    out_size: int,
    points: int,
) -> Batch:
    """Turn a generated pair into training tensors.

    Args:
        pair: Generated shape pair.
        out_size: Decoder output side used for heatmap targets.
        points: Number of boundary points for the descriptor loss.

    Returns:
        A batched sample.
    """
    heat0 = boundary_target(pair.mask0, out_size=out_size)
    heat1 = boundary_target(pair.mask1, out_size=out_size)
    samples0 = _resample_polyline(pair.points0, points)
    samples1 = _project(pair.homography, samples0).astype(np.float32)

    centre = samples0.mean(axis=0, keepdims=True)
    span = float(np.ptp(samples0, axis=0).max()) or 1.0
    normalised = (samples0 - centre) / span
    symmetry = symmetry_mask(normalised, threshold=0.12).numpy()
    return Batch(
        image0=pair.image0,
        image1=pair.image1,
        heat0=heat0,
        heat1=heat1,
        points0=samples0,
        points1=samples1,
        symmetry=symmetry,
    )


def load_checkpoint(
    path: Path,
    *,
    device: str | None = None,
) -> DetectorDescriptor:
    """Load trained heads into a fresh detector.

    Args:
        path: Checkpoint written by :func:`train`.
        device: Device override.

    Returns:
        A detector with the trained decoder and heads.
    """
    payload = torch.load(path, map_location="cpu")
    config = ModelConfig(**payload["config"])
    model = DetectorDescriptor(config, device=device)
    model.decoder.load_state_dict(payload["decoder"])
    model.boundary.load_state_dict(payload["boundary"])
    model.descriptor.load_state_dict(payload["descriptor"])
    model.eval()
    return model


def _sample_descriptors(
    descriptor_map: torch.Tensor,
    points: np.ndarray,
    size: int,
) -> torch.Tensor:
    """Bilinearly sample descriptors at image coordinates.

    Args:
        descriptor_map: ``(1, D, h, w)`` descriptors.
        points: ``(K, 2)`` coordinates in the model input frame.
        size: Side of the model input frame.

    Returns:
        ``(K, D)`` unit-length descriptors.
    """
    coordinates = torch.as_tensor(
        points, dtype=torch.float32, device=descriptor_map.device
    )
    coordinates = coordinates / max(size - 1, 1) * 2.0 - 1.0
    grid = coordinates.view(1, 1, -1, 2)
    sampled = F.grid_sample(
        descriptor_map,
        grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return F.normalize(sampled[0, :, 0].T, dim=1)


def train(args: argparse.Namespace) -> int:
    """Run the training loop.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    config = ModelConfig(input_size=args.size, top_k=args.max_keypoints)
    model = DetectorDescriptor(config, device=str(device))
    model.train()

    parameters = [
        parameter
        for module in (model.decoder, model.boundary, model.descriptor)
        for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)

    train_generator = ShapePairGenerator(size=args.size, limit=10_000, seed=args.seed)
    validation_generator = ShapePairGenerator(
        size=args.size,
        limit=10_000,
        seed=args.seed + 101,
    )
    out_size = args.size // 4

    print(
        f"device {device} | trainable {sum(p.numel() for p in parameters) / 1e3:.1f}k "
        f"| steps {args.steps} | batch {args.batch}"
    )
    started = time.time()
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        heat_total = torch.zeros((), device=device)
        descriptor_total = torch.zeros((), device=device)

        for offset in range(args.batch):
            index = (step - 1) * args.batch + offset
            batch = build_batch(
                train_generator[index],
                out_size=out_size,
                points=args.points,
            )
            logits0, descriptors0 = model.maps(batch.image0)
            logits1, descriptors1 = model.maps(batch.image1)

            heat = focal_loss(
                logits0, torch.as_tensor(batch.heat0, device=device)[None]
            )
            heat = heat + focal_loss(
                logits1, torch.as_tensor(batch.heat1, device=device)[None]
            )
            sampled0 = _sample_descriptors(descriptors0, batch.points0, args.size)
            sampled1 = _sample_descriptors(descriptors1, batch.points1, args.size)
            ignore = torch.as_tensor(batch.symmetry, device=device)
            descriptor = infonce(
                sampled0,
                sampled1,
                temperature=args.temperature,
                ignore=ignore,
            )

            heat_total = heat_total + heat / args.batch
            descriptor_total = descriptor_total + descriptor / args.batch

        loss = heat_total + args.descriptor_weight * descriptor_total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            metrics = _evaluate(model, validation_generator, out_size, args, device)
            history.append({"step": float(step), **metrics})
            print(
                f"step {step:4d} | heat {heat_total.item():.3f} "
                f"| desc {descriptor_total.item():.3f} "
                f"| val desc {metrics['descriptor']:.2f} "
                f"vs chance {np.log(args.points):.2f} "
                f"| top1 {metrics['top1']:.3f} (chance {metrics['chance']:.3f}) "
                f"| ignored {metrics['ignored']:.0%} "
                f"| P {metrics['precision']:.2f} R {metrics['recall']:.2f} "
                f"| kp {metrics['peaks']:.0f} "
                f"| {time.time() - started:.0f}s"
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                _save(model, config, args.output / "best.pt")

    _save(model, config, args.output / "heads.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"wrote {args.output / 'heads.pt'} (best validation F1 {best_f1:.3f})")
    return 0


@torch.no_grad()
def _evaluate(
    model: DetectorDescriptor,
    generator: ShapePairGenerator,
    out_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on a held-out seed range.

    Reports what tells "the model got better" apart from "the loss went down":
    heatmap precision and recall at the threshold inference actually uses, on
    both views; descriptor top-1 accuracy against its chance floor; the share of
    negatives the symmetry mask suppresses; and how many keypoints the detector
    keeps.

    Args:
        model: Model under training.
        generator: Validation generator.
        out_size: Decoder output side.
        args: Parsed arguments.
        device: Device under test.

    Returns:
        Loss, accuracy, and coverage metrics.
    """
    model.eval()
    heats: list[float] = []
    descriptors: list[float] = []
    top1: list[float] = []
    ignored: list[float] = []
    peaks: list[float] = []
    true_positive = false_positive = false_negative = 0
    threshold = model.config.boundary_threshold

    for index in range(args.validation_samples):
        batch = build_batch(generator[index], out_size=out_size, points=args.points)
        logits0, map0 = model.maps(batch.image0)
        logits1, map1 = model.maps(batch.image1)
        target0 = torch.as_tensor(batch.heat0, device=device)[None]
        target1 = torch.as_tensor(batch.heat1, device=device)[None]
        heats.append(float(focal_loss(logits0, target0) + focal_loss(logits1, target1)))

        sampled0 = _sample_descriptors(map0, batch.points0, args.size)
        sampled1 = _sample_descriptors(map1, batch.points1, args.size)
        ignore = torch.as_tensor(batch.symmetry, device=device)
        descriptors.append(
            float(
                infonce(sampled0, sampled1, temperature=args.temperature, ignore=ignore)
            )
        )

        predictions = (sampled0 @ sampled1.T).argmax(dim=1)
        labels = torch.arange(len(predictions), device=device)
        top1.append(float((predictions == labels).float().mean()))
        off_diagonal = max(len(batch.symmetry) * (len(batch.symmetry) - 1), 1)
        ignored.append(float(batch.symmetry.sum()) / off_diagonal)

        for logits, target in ((logits0, target0), (logits1, target1)):
            predicted = torch.sigmoid(logits)[0, 0] >= threshold
            truth = target[0, 0] >= 0.5
            true_positive += int((predicted & truth).sum())
            false_positive += int((predicted & ~truth).sum())
            false_negative += int((~predicted & truth).sum())
            points, _ = extract_keypoints(
                logits,
                threshold=threshold,
                top_k=model.config.top_k,
            )
            peaks.append(float(len(points)))

    model.train()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "heat": float(np.mean(heats)),
        "descriptor": float(np.mean(descriptors)),
        "top1": float(np.mean(top1)),
        "chance": 1.0 / max(args.points, 1),
        "ignored": float(np.mean(ignored)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "peaks": float(np.median(peaks)),
    }


def _save(model: DetectorDescriptor, config: ModelConfig, path: Path) -> None:
    """Write the trained heads and the architecture config.

    Args:
        model: Model to save.
        config: Architecture settings.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.__dict__,
            "decoder": model.decoder.state_dict(),
            "boundary": model.boundary.state_dict(),
            "descriptor": model.descriptor.state_dict(),
        },
        path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the boundary heads.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--descriptor-weight", type=float, default=1.0)
    parser.add_argument("--validation-samples", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/train"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and train.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    return train(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
