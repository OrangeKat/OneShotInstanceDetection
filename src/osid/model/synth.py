"""Synthetic pairs of shapes on flat backgrounds, with distractors.

Control is the point: a filled shape on a single-colour background, with the
number of sides, size, rotation, perspective, contrast, blur and noise set
independently per view. Distractors — other shapes in the same frame — are what
make the task about *this* shape rather than about edges in general.

Everything is exact by construction: the polygon transform is the
correspondence, and the rasterised polygon is the mask.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "ShapePair",
    "ShapePairGenerator",
    "boundary_band",
]

SUPERSAMPLE = 2
SHAPE_KINDS = ("polygon", "rounded", "blob", "star", "ellipse")


@dataclass
class ShapePair:
    """One generated training pair.

    Attributes:
        image0: First view, ``(size, size, 3)`` uint8 RGB.
        image1: Second view.
        mask0: Object silhouette in view 0 (the reference contains no
            distractors), uint8 0/1.
        mask1: Silhouette of every shape in view 1, so the heatmap target
            covers the clutter the detector should also fire on.
        object_mask0: Object-only silhouette in view 0, for label checks.
        object_mask1: Object-only silhouette in view 1.
        homography: ``3 x 3`` map from view 0 to view 1.
        points0: Object polygon vertices in view 0.
        points1: Object polygon vertices in view 1.
        negatives1: ``(M, 2)`` distractor boundary points in view 1, used as
            extra negatives in the descriptor loss.
        meta: The parameters used, for slicing results by difficulty.
    """

    image0: np.ndarray
    image1: np.ndarray
    mask0: np.ndarray
    mask1: np.ndarray
    object_mask0: np.ndarray
    object_mask1: np.ndarray
    homography: np.ndarray
    points0: np.ndarray
    points1: np.ndarray
    negatives1: np.ndarray
    meta: dict[str, float]


def _normalise(points: np.ndarray) -> np.ndarray:
    """Scale points so the largest radius is one.

    Args:
        points: ``(N, 2)`` boundary points.

    Returns:
        ``(N, 2)`` points with maximum radius 1.
    """
    radius = float(np.linalg.norm(points, axis=1).max())
    return points / max(radius, 1e-6)


def _chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Round the corners of a closed polygon.

    Args:
        points: ``(N, 2)`` polygon vertices.
        iterations: Corner-cutting passes.

    Returns:
        A denser, smoother closed boundary.
    """
    smoothed = points
    for _ in range(iterations):
        shifted = np.roll(smoothed, -1, axis=0)
        first = 0.75 * smoothed + 0.25 * shifted
        second = 0.25 * smoothed + 0.75 * shifted
        smoothed = np.stack([first, second], axis=1).reshape(-1, 2)
    return smoothed


class ShapePairGenerator:
    """Generates shape pairs with controllable difficulty.

    Attributes:
        size: Canvas side in pixels.
        limit: Samples per epoch.
        distractors: Extra shapes placed around the object.
        negative_points: Distractor boundary points kept per pair.
    """

    def __init__(
        self,
        *,
        size: int = 448,
        limit: int = 4000,
        seed: int = 0,
        contrast_range: tuple[float, float] = (0.06, 0.45),
        blur_range: tuple[float, float] = (0.0, 2.0),
        noise_range: tuple[float, float] = (0.0, 0.03),
        scale_range: tuple[float, float] = (0.08, 0.5),
        same_background: bool = False,
        shape_kind: str | None = None,
        distractors: int = 2,
        negative_points: int = 64,
    ) -> None:
        """Configure the generator.

        Args:
            size: Square canvas side.
            limit: Samples per epoch.
            seed: Base seed for reproducibility.
            contrast_range: Luminance difference between shape and background.
            blur_range: Per-view Gaussian blur sigma.
            noise_range: Per-view additive noise sigma.
            scale_range: Polygon radius as a fraction of the canvas.
            same_background: Share one background colour across the views.
            shape_kind: Force one family, or ``None`` to sample them.
            distractors: Extra shapes per frame, 0 for the object alone.
            negative_points: Distractor boundary points returned for the loss.
        """
        if shape_kind is not None and shape_kind not in SHAPE_KINDS:
            raise ValueError(f"unknown shape kind: {shape_kind}")
        self.size = size
        self.limit = limit
        self.seed = seed
        self.contrast_range = contrast_range
        self.blur_range = blur_range
        self.noise_range = noise_range
        self.scale_range = scale_range
        self.same_background = same_background
        self.shape_kind = shape_kind
        self.distractors = distractors
        self.negative_points = negative_points

    def __len__(self) -> int:
        """Return the number of samples per epoch.

        Returns:
            Sample count.
        """
        return self.limit

    def __getitem__(self, index: int) -> ShapePair:
        """Generate one pair.

        Args:
            index: Sample index; seeds the RNG so a sample is reproducible.

        Returns:
            A generated pair.
        """
        rng = np.random.default_rng(int(index) + self.seed * 104_729 + 17)
        object_polygon, kind = self._shape(rng)
        shapes = [
            {
                "polygon": object_polygon,
                "transform0": self._placement(object_polygon, rng),
                "transform1": self._placement(object_polygon, rng),
                "contrast0": float(rng.uniform(*self.contrast_range)),
                "contrast1": float(rng.uniform(*self.contrast_range)),
            }
        ]
        for _ in range(self.distractors):
            polygon, _ = self._shape(rng)
            shapes.append(
                {
                    "polygon": polygon,
                    "transform0": self._placement(polygon, rng),
                    "transform1": self._placement(polygon, rng),
                    "contrast0": float(rng.uniform(*self.contrast_range)),
                    "contrast1": float(rng.uniform(*self.contrast_range)),
                }
            )

        background0 = self._colour(rng)
        background1 = background0 if self.same_background else self._colour(rng)
        blur0 = float(rng.uniform(*self.blur_range))
        blur1 = float(rng.uniform(*self.blur_range))
        noise0 = float(rng.uniform(*self.noise_range))
        noise1 = float(rng.uniform(*self.noise_range))

        image0, mask0, object0 = self._view(
            shapes[:1], 0, background0, blur0, noise0, rng
        )
        image1, mask1, object1 = self._view(shapes, 1, background1, blur1, noise1, rng)

        object_shape = shapes[0]
        points0 = self._apply(object_shape["transform0"], object_polygon)
        points1 = self._apply(object_shape["transform1"], object_polygon)
        homography = object_shape["transform1"] @ np.linalg.inv(
            object_shape["transform0"]
        )
        negatives = self._distractor_points(shapes[1:], "transform1", rng)

        meta = {
            "kind": float(SHAPE_KINDS.index(kind)),
            "sides": float(len(object_polygon)),
            "contrast0": object_shape["contrast0"],
            "contrast1": object_shape["contrast1"],
            "blur0": blur0,
            "blur1": blur1,
            "noise0": noise0,
            "noise1": noise1,
            "distractors": float(len(shapes) - 1),
        }
        return ShapePair(
            image0=image0,
            image1=image1,
            mask0=mask0,
            mask1=mask1,
            object_mask0=object0,
            object_mask1=object1,
            homography=homography.astype(np.float64),
            points0=points0,
            points1=points1,
            negatives1=negatives,
            meta=meta,
        )

    # ------------------------------------------------------------------ shapes
    def _shape(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        """Sample a closed silhouette in canonical coordinates.

        Args:
            rng: Random generator.

        Returns:
            ``(N, 2)`` points on the boundary and the family name.
        """
        kind = (
            self.shape_kind
            if self.shape_kind is not None
            else SHAPE_KINDS[int(rng.integers(len(SHAPE_KINDS)))]
        )
        if kind == "polygon":
            sides = int(rng.integers(3, 11))
            angles = np.sort(rng.uniform(0, 2 * np.pi, sides))
            radii = rng.uniform(0.75, 1.0, sides)
            points = np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
            return _normalise(points), kind

        if kind == "rounded":
            sides = int(rng.integers(3, 9))
            angles = np.sort(rng.uniform(0, 2 * np.pi, sides))
            radii = rng.uniform(0.8, 1.0, sides)
            points = np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
            return _normalise(_chaikin(points, iterations=2)), kind

        if kind == "blob":
            theta = np.linspace(0, 2 * np.pi, 160, endpoint=False)
            radius = np.ones_like(theta)
            for harmonic in range(2, int(rng.integers(4, 7))):
                amplitude = float(rng.uniform(0.04, 0.22))
                phase = float(rng.uniform(0, 2 * np.pi))
                radius += amplitude * np.cos(harmonic * theta + phase)
            radius = np.clip(radius, 0.25, None)
            points = np.stack([np.cos(theta) * radius, np.sin(theta) * radius], axis=1)
            return _normalise(points), kind

        if kind == "star":
            lobes = int(rng.integers(4, 9))
            theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
            depth = float(rng.uniform(0.35, 0.6))
            radius = 1.0 - depth * (0.5 + 0.5 * np.cos(lobes * theta))
            points = np.stack([np.cos(theta) * radius, np.sin(theta) * radius], axis=1)
            return _normalise(points), kind

        theta = np.linspace(0, 2 * np.pi, 160, endpoint=False)
        aspect = float(rng.uniform(0.35, 1.0))
        points = np.stack([np.cos(theta), np.sin(theta) * aspect], axis=1)
        if rng.random() < 0.5:
            points = points[:, ::-1]
        return _normalise(points), kind

    def _colour(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a background colour.

        Args:
            rng: Random generator.

        Returns:
            RGB colour in ``[0, 1]``.
        """
        return rng.uniform(0.15, 0.85, 3).astype(np.float32)

    def _placement(
        self,
        polygon: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a placement transform for one shape.

        Args:
            polygon: Canonical polygon vertices.
            rng: Random generator.

        Returns:
            ``3 x 3`` transform from canonical coordinates to the canvas.
        """
        radius = float(rng.uniform(*self.scale_range)) * self.size
        angle = float(rng.uniform(-180.0, 180.0))
        max_norm = float(np.linalg.norm(polygon, axis=1).max())
        radius = min(radius, (self.size / 2.0 - 3.0) / max(max_norm, 1e-6))
        extent = 1.08 * radius * max_norm
        low = extent + 2.0
        high = self.size - extent - 2.0
        centre = (
            np.array([self.size / 2.0, self.size / 2.0])
            if low >= high
            else rng.uniform(low, high, 2)
        )
        radians = np.deg2rad(angle)
        rotation = np.array(
            [
                [np.cos(radians), -np.sin(radians), 0.0],
                [np.sin(radians), np.cos(radians), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scaling = np.diag([radius, radius, 1.0])
        perspective = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [rng.normal(0.0, 8e-5), rng.normal(0.0, 8e-5), 1.0],
            ]
        )
        translation = np.array(
            [[1.0, 0.0, centre[0]], [0.0, 1.0, centre[1]], [0.0, 0.0, 1.0]]
        )
        return translation @ perspective @ rotation @ scaling

    def _view(
        self,
        shapes: list[dict[str, object]],
        view: int,
        background: np.ndarray,
        blur: float,
        noise: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render every shape onto one background.

        Args:
            shapes: Shape records with polygons, transforms and contrasts.
            view: 0 or 1, selecting the transform and contrast to use.
            background: Background colour in ``[0, 1]``.
            blur: Gaussian sigma applied after compositing.
            noise: Additive Gaussian sigma.
            rng: Random generator, used for the contrast direction.

        Returns:
            RGB uint8 image, the union silhouette, and the object silhouette.
        """
        image = np.repeat(background[None, None, :], self.size, axis=0)
        image = np.repeat(image, self.size, axis=1)
        union = np.zeros((self.size, self.size), dtype=np.float32)
        object_mask = np.zeros((self.size, self.size), dtype=np.float32)
        for position, shape in enumerate(shapes):
            polygon = shape["polygon"]
            transform = shape[f"transform{view}"]
            contrast = float(shape[f"contrast{view}"])
            alpha = self._raster(polygon, transform)
            direction = rng.normal(0.0, 1.0, 3)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            foreground = np.clip(
                background + direction.astype(np.float32) * contrast, 0.0, 1.0
            )
            image = image * (1.0 - alpha[:, :, None]) + foreground * alpha[:, :, None]
            union = np.maximum(union, alpha)
            if position == 0:
                object_mask = alpha
        if blur > 0:
            image = cv2.GaussianBlur(image, (0, 0), blur)
        if noise > 0:
            image = image + rng.normal(0.0, noise, image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return (
            (image * 255.0).astype(np.uint8),
            (union > 0.5).astype(np.uint8),
            (object_mask > 0.5).astype(np.uint8),
        )

    def _raster(
        self,
        polygon: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """Rasterise one transformed polygon into an alpha map.

        Args:
            polygon: Canonical polygon vertices.
            transform: Canonical-to-canvas transform.

        Returns:
            ``(size, size)`` alpha map in ``[0, 1]``.
        """
        points = self._apply(transform, polygon)
        high = self.size * SUPERSAMPLE
        canvas = np.zeros((high, high), dtype=np.uint8)
        cv2.fillPoly(canvas, [np.round(points * SUPERSAMPLE).astype(np.int32)], 255)
        alpha = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return alpha.astype(np.float32) / 255.0

    def _distractor_points(
        self,
        shapes: list[dict[str, object]],
        transform_key: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample boundary points on the distractor shapes.

        Args:
            shapes: Distractor records.
            transform_key: ``transform0`` or ``transform1``.
            rng: Random generator.

        Returns:
            ``(M, 2)`` points in the requested view.
        """
        if not shapes or self.negative_points <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        per_shape = max(1, self.negative_points // len(shapes))
        samples = []
        for shape in shapes:
            points = self._apply(shape[transform_key], shape["polygon"])
            closed = np.concatenate([points, points[:1]], axis=0)
            cumulative = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
            )
            total = float(cumulative[-1])
            if total <= 0:
                continue
            for position in np.linspace(0.0, total, per_shape, endpoint=False):
                index = min(
                    int(np.searchsorted(cumulative, position, side="right") - 1),
                    len(points) - 1,
                )
                local = (position - cumulative[index]) / max(
                    cumulative[index + 1] - cumulative[index], 1e-9
                )
                samples.append(
                    closed[index] + local * (closed[index + 1] - closed[index])
                )
        return (
            np.asarray(samples, dtype=np.float32)
            if samples
            else np.zeros((0, 2), dtype=np.float32)
        )

    @staticmethod
    def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply a homography to 2-D points.

        Args:
            transform: ``3 x 3`` matrix.
            points: ``(N, 2)`` points.

        Returns:
            ``(N, 2)`` transformed points.
        """
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        projected = homogeneous @ transform.T
        return projected[:, :2] / projected[:, 2:3]


def boundary_band(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Rasterise a boundary band from a binary mask.

    Args:
        mask: Binary silhouette, ``(H, W)`` uint8 0/1.
        width: Band half-width in pixels.

    Returns:
        Binary band, ``(H, W)`` uint8 0/1.
    """
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
    if width > 1:
        edge = cv2.dilate(edge, np.ones((width, width), np.uint8))
    return (edge > 0).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    """Render a contact sheet, a difficulty grid, or check label alignment.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Preview synthetic shape pairs.")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/synthetic-shapes.png"),
    )
    args = parser.parse_args(argv)

    generator = ShapePairGenerator(
        size=args.size,
        limit=args.samples,
        seed=args.seed,
        distractors=args.distractors,
    )
    if args.check:
        ious, errors = [], []
        for index in range(args.samples):
            pair = generator[index]
            warped = cv2.warpPerspective(
                pair.object_mask0,
                pair.homography,
                (args.size, args.size),
                flags=cv2.INTER_NEAREST,
            )
            intersection = int(np.logical_and(warped > 0, pair.object_mask1 > 0).sum())
            union = int(np.logical_or(warped > 0, pair.object_mask1 > 0).sum())
            ious.append(intersection / union if union else 0.0)
            projected = ShapePairGenerator._apply(pair.homography, pair.points0)
            errors.append(float(np.abs(projected - pair.points1).max()))
        print(f"mask alignment IoU: mean {np.mean(ious):.3f} min {np.min(ious):.3f}")
        print(f"vertex reprojection error: max {max(errors):.2e} px")

    if args.grid:
        _write_grid(args.output, size=args.size // 2)
        return 0

    cells = []
    for index in range(args.samples):
        pair = generator[index]
        cells.append(
            np.concatenate(
                [
                    _annotate(pair.image0, pair.mask0, pair.meta, "A"),
                    _annotate(pair.image1, pair.mask1, pair.meta, "B"),
                ],
                axis=1,
            )
        )
    sheet = np.concatenate(cells, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.output}")
    return 0


def _write_grid(path: Path, *, size: int) -> None:
    """Render a grid where only contrast and blur change.

    Args:
        path: Output image path.
        size: Cell side in pixels.
    """
    contrasts = [0.04, 0.10, 0.22, 0.40]
    blurs = [0.0, 1.2, 2.4]
    rows = []
    for blur in blurs:
        cells = []
        for contrast in contrasts:
            generator = ShapePairGenerator(
                size=size,
                limit=1,
                seed=7,
                contrast_range=(contrast, contrast),
                blur_range=(blur, blur),
                noise_range=(0.015, 0.015),
                same_background=True,
                shape_kind="blob",
                distractors=0,
            )
            pair = generator[0]
            cell = pair.image1.copy()
            contours, _ = cv2.findContours(
                pair.mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(cell, contours, -1, (60, 220, 90), 1)
            cv2.putText(
                cell,
                f"contrast {contrast:.2f}  blur {blur:.1f}",
                (6, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
            )
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {path} ({len(contrasts)}x{len(blurs)} grid)")


def _annotate(
    image: np.ndarray,
    mask: np.ndarray,
    meta: dict[str, float],
    label: str,
) -> np.ndarray:
    """Draw the silhouettes and the difficulty knobs.

    Args:
        image: RGB image.
        mask: Binary silhouette.
        meta: Difficulty settings.
        label: ``A`` or ``B``.

    Returns:
        Annotated RGB image.
    """
    view = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (60, 220, 90), 1)
    suffix = "0" if label == "A" else "1"
    text = (
        f"{label}  contrast {meta[f'contrast{suffix}']:.2f}  "
        f"blur {meta[f'blur{suffix}']:.1f}  noise {meta[f'noise{suffix}']:.3f}  "
        f"distractors {int(meta['distractors'])}"
    )
    cv2.putText(view, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return view


if __name__ == "__main__":
    raise SystemExit(main())
