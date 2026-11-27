"""Synthetic pairs: one object pasted into two different backgrounds.

Training a detector/descriptor needs three things the benchmark data does not
give: a mask per image, exact correspondences between two views, and enough
pairs to learn from. Pasting a reference crop into random backgrounds produces
all three exactly — the transform is known, so it *is* the correspondence.

Each sample is:

- view A: an object crop scaled, rotated, and pasted into background A;
- view B: the same object pasted into background B with a different transform;
- ``homography``: the map from A to B, valid on the object;
- ``mask``: the object's silhouette in each view, with occlusions removed.

Backgrounds differ between the views on purpose: otherwise the model can match
background instead of the object.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from osid.utils.dataset import InstanceDataset

__all__ = [
    "SyntheticPair",
    "SyntheticPairGenerator",
    "boundary_band",
]


@dataclass
class SyntheticPair:
    """One generated training pair.

    Attributes:
        image0: First view, ``(size, size, 3)`` uint8 RGB.
        image1: Second view.
        mask0: Object silhouette in view 0, uint8 0/1.
        mask1: Object silhouette in view 1, occlusions removed.
        homography: ``3 x 3`` map from view 0 to view 1.
    """

    image0: np.ndarray
    image1: np.ndarray
    mask0: np.ndarray
    mask1: np.ndarray
    homography: np.ndarray


class SyntheticPairGenerator:
    """Generates object-in-background pairs from a manifest of crops.

    Attributes:
        size: Canvas side in pixels.
        limit: Maximum samples per epoch.
        rng: Random generator, seeded for reproducibility.
    """

    def __init__(
        self,
        manifest: Path,
        *,
        size: int = 448,
        limit: int = 4000,
        seed: int = 0,
        background_pool: int = 64,
        occlusion_probability: float = 0.35,
    ) -> None:
        """Load the object crops and a small pool of backgrounds.

        Args:
            manifest: Dataset manifest used for crops and backgrounds.
            size: Square canvas side.
            limit: Samples per epoch.
            seed: Seed for the generator.
            background_pool: Number of images kept in memory as backgrounds.
            occlusion_probability: Chance of covering part of the object.
        """
        self.dataset = InstanceDataset(manifest, load_images=False)
        self.size = size
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.objects = [s for s in self.dataset.samples if s.reference_path.exists()]
        if not self.objects:
            raise ValueError(f"no reference crops found via {manifest}")
        self.backgrounds = self._load_backgrounds(background_pool)
        self.occlusion_probability = occlusion_probability

    def __len__(self) -> int:
        """Return the number of samples per epoch.

        Returns:
            Sample count.
        """
        return self.limit

    def __getitem__(self, index: int) -> SyntheticPair:
        """Generate one pair.

        Args:
            index: Sample index, used to seed the RNG so the sample is
                reproducible.

        Returns:
            A generated pair.
        """
        rng = np.random.default_rng(int(index) + 7919)
        sample = self.objects[int(rng.integers(len(self.objects)))]
        object_rgb, object_mask = self._load_object(sample, rng)

        transform0 = self._place_transform(object_rgb.shape[:2], rng)
        transform1 = self._place_transform(object_rgb.shape[:2], rng)
        image0, mask0 = self._composite(object_rgb, object_mask, transform0, rng)
        image1, mask1 = self._composite(object_rgb, object_mask, transform1, rng)

        homography = transform1 @ np.linalg.inv(transform0)
        image0 = self._degrade(image0, rng)
        image1 = self._degrade(image1, rng)
        return SyntheticPair(
            image0=image0,
            image1=image1,
            mask0=mask0,
            mask1=mask1,
            homography=homography.astype(np.float64),
        )

    def _load_backgrounds(self, count: int) -> list[np.ndarray]:
        """Load a pool of background images.

        Args:
            count: How many images to keep.

        Returns:
            RGB images.
        """
        pool: list[np.ndarray] = []
        for sample in self.objects[:count]:
            image = cv2.imread(str(sample.query_path), cv2.IMREAD_COLOR)
            if image is not None:
                pool.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not pool:
            pool.append(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        return pool

    def _load_object(
        self,
        sample: object,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load one reference crop and its silhouette.

        Args:
            sample: Dataset sample.
            rng: Random generator, used to jitter the crop border.

        Returns:
            RGB crop and a uint8 silhouette.
        """
        loaded = self.dataset.load(sample)  # type: ignore[arg-type]
        crop = self.dataset.reference_crop(loaded)
        mask = np.ones(crop.shape[:2], dtype=np.uint8)
        if rng.random() < 0.5:
            margin = int(0.04 * min(crop.shape[:2]))
            if margin > 1:
                mask[:margin, :] = 0
                mask[-margin:, :] = 0
                mask[:, :margin] = 0
                mask[:, -margin:] = 0
        return crop, mask

    def _place_transform(
        self,
        crop_shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a placement transform for one view.

        The crop is scaled so its long side covers a fraction of the canvas,
        rotated about its own centre, then moved onto the canvas.

        Args:
            crop_shape: ``(height, width)`` of the object crop.
            rng: Random generator.

        Returns:
            ``3 x 3`` transform from crop coordinates to canvas coordinates.
        """
        height, width = crop_shape
        long_side = max(height, width)
        target = float(rng.uniform(0.2, 0.5)) * self.size
        scale = target / long_side
        angle = float(rng.uniform(-25.0, 25.0))
        # Keep the rotated crop fully inside the canvas, so the two views show
        # the same object extent and the correspondence labels stay exact.
        scaled_w = width * scale
        scaled_h = height * scale
        half_diag = 0.5 * float(np.hypot(scaled_w, scaled_h))
        low = half_diag + 2.0
        high = self.size - half_diag - 2.0
        if low >= high:
            centre = (self.size / 2.0, self.size / 2.0)
        else:
            centre = (
                float(rng.uniform(low, high)),
                float(rng.uniform(low, high)),
            )
        to_origin = np.array(
            [[1.0, 0.0, -width / 2], [0.0, 1.0, -height / 2], [0.0, 0.0, 1.0]]
        )
        scaling = np.diag([scale, scale, 1.0])
        radians = np.deg2rad(angle)
        rotation = np.array(
            [
                [np.cos(radians), -np.sin(radians), 0.0],
                [np.sin(radians), np.cos(radians), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        to_centre = np.array(
            [[1.0, 0.0, centre[0]], [0.0, 1.0, centre[1]], [0.0, 0.0, 1.0]]
        )
        perspective = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [rng.normal(0.0, 4e-5), rng.normal(0.0, 4e-5), 1.0],
            ]
        )
        return to_centre @ perspective @ rotation @ scaling @ to_origin

    def _composite(
        self,
        crop: np.ndarray,
        mask: np.ndarray,
        transform: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Paste the object into a random background.

        Args:
            crop: Object RGB crop.
            mask: Object silhouette.
            transform: Crop-to-canvas transform.
            rng: Random generator.

        Returns:
            The composed RGB image and the object silhouette on the canvas.
        """
        source = crop.astype(np.float32)
        source = source / 255.0
        normalised = np.zeros_like(source)
        normalised[:] = source
        warped = cv2.warpPerspective(
            normalised,
            transform,
            (self.size, self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_mask = cv2.warpPerspective(
            mask,
            transform,
            (self.size, self.size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        background = self._background(rng)
        feather = cv2.GaussianBlur(warped_mask.astype(np.float32), (0, 0), 1.2)[
            :, :, None
        ]
        composed = warped * feather + background * (1.0 - feather)
        composed = np.clip(composed, 0.0, 1.0)
        image = (composed * 255.0).astype(np.uint8)

        occlusion = self._occlusion(transform, mask.shape, rng)
        if occlusion is not None:
            texture = cv2.GaussianBlur(image, (0, 0), 9)
            image[occlusion] = texture[occlusion]
            warped_mask[occlusion] = 0
        return image, warped_mask

    def _background(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random background patch in ``[0, 1]``.

        Args:
            rng: Random generator.

        Returns:
            ``(size, size, 3)`` float image.
        """
        if rng.random() < 0.15:
            base = rng.normal(0.5, 0.08, (self.size, self.size, 1))
            tint = rng.normal(0.0, 0.03, (1, 1, 3))
            noise = np.clip(base + tint, 0.0, 1.0)
            return noise.astype(np.float32)
        image = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
        height, width = image.shape[:2]
        side = min(height, width)
        top = int(rng.integers(0, max(1, height - side + 1)))
        left = int(rng.integers(0, max(1, width - side + 1)))
        patch = image[top : top + side, left : left + side]
        patch = cv2.resize(patch, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return patch.astype(np.float32) / 255.0

    def _occlusion(
        self,
        transform: np.ndarray,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Return a boolean mask of an occluding rectangle, or ``None``.

        Args:
            transform: Object placement, used to place the occluder on the object.
            shape: Object crop shape.
            rng: Random generator.

        Returns:
            Boolean mask of the occluded pixels, or ``None``.
        """
        if rng.random() > self.occlusion_probability:
            return None
        height, width = shape
        x0 = int(rng.integers(0, max(1, width // 2)))
        y0 = int(rng.integers(0, max(1, height // 2)))
        x1 = int(min(width, x0 + rng.integers(width // 4, max(width // 4 + 1, width))))
        y1 = int(
            min(height, y0 + rng.integers(height // 4, max(height // 4 + 1, height)))
        )
        if x1 <= x0 or y1 <= y0:
            return None
        patch = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
        warped = cv2.warpPerspective(
            patch,
            transform,
            (self.size, self.size),
            flags=cv2.INTER_NEAREST,
        )
        return warped.astype(bool)

    def _degrade(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply photometric and sensor noise.

        Args:
            image: RGB uint8 image.
            rng: Random generator.

        Returns:
            Degraded RGB uint8 image.
        """
        float_image = image.astype(np.float32) / 255.0
        float_image *= float(rng.uniform(0.75, 1.25))
        float_image += float(rng.uniform(-0.08, 0.08))
        float_image = np.clip(float_image, 0.0, 1.0)
        float_image = float_image ** float(rng.uniform(0.85, 1.15))

        if rng.random() < 0.35:
            sigma = float(rng.uniform(0.4, 1.6))
            float_image = cv2.GaussianBlur(float_image, (0, 0), sigma)
        if rng.random() < 0.35:
            float_image += rng.normal(0.0, rng.uniform(0.005, 0.03), float_image.shape)
        float_image = np.clip(float_image, 0.0, 1.0)
        image = (float_image * 255.0).astype(np.uint8)

        if rng.random() < 0.3:
            quality = int(rng.integers(35, 90))
            ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if ok:
                image = cv2.cvtColor(
                    cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
        return image


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
    """Generate a few samples and save a contact sheet.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Preview synthetic pairs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/flickrlogos32/manifest.json"),
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report mask alignment under the generated homographies.",
    )
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/synthetic-preview.png"),
    )
    args = parser.parse_args(argv)

    generator = SyntheticPairGenerator(
        args.manifest,
        size=args.size,
        limit=args.samples,
        occlusion_probability=0.0 if args.check else 0.35,
    )
    rows = []
    for index in range(args.samples):
        pair = generator[index]
        rows.append(
            np.concatenate(
                [
                    _annotate(pair.image0, pair.mask0),
                    _annotate(pair.image1, pair.mask1),
                ],
                axis=1,
            )
        )
    if args.check:
        ious = []
        for index in range(args.samples):
            pair = generator[index]
            warped = cv2.warpPerspective(
                pair.mask0,
                pair.homography,
                (pair.image0.shape[1], pair.image1.shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            intersection = int(np.logical_and(warped > 0, pair.mask1 > 0).sum())
            union = int(np.logical_or(warped > 0, pair.mask1 > 0).sum())
            ious.append(intersection / union if union else 0.0)
        print(f"mask alignment IoU: mean {np.mean(ious):.3f} min {np.min(ious):.3f}")

    sheet = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.output} ({len(rows)} pairs, {args.size}px)")
    return 0


def _annotate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Outline the object silhouette for the preview.

    Args:
        image: RGB image.
        mask: Binary silhouette.

    Returns:
        Annotated RGB image.
    """
    view = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (255, 90, 60), 1)
    return view


if __name__ == "__main__":
    raise SystemExit(main())
