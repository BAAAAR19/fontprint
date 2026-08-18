"""Image normalization and OCR-free typographic region proposals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def otsu_threshold(gray: np.ndarray) -> int:
    """Compute an Otsu threshold using only NumPy."""

    values = np.asarray(gray, dtype=np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = values.size
    if total == 0:
        return 127
    probabilities = histogram / total
    omega = np.cumsum(probabilities)
    means = np.cumsum(probabilities * np.arange(256))
    global_mean = means[-1]
    denominator = omega * (1.0 - omega)
    variance = np.zeros_like(denominator)
    valid = denominator > 1e-12
    variance[valid] = (global_mean * omega[valid] - means[valid]) ** 2 / denominator[valid]
    return int(np.argmax(variance))


def ink_mask(image: Image.Image) -> np.ndarray:
    """Return a binary dark-ink mask; handles white or transparent backgrounds."""

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = min(245, max(40, otsu_threshold(gray)))
    mask = gray <= threshold
    if mask.mean() > 0.55:
        mask = ~mask
    return mask


def crop_to_ink(image: Image.Image, padding: int = 3) -> Image.Image:
    mask = ink_mask(image)
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        return image.convert("L")
    y1, x1 = coordinates.min(axis=0)
    y2, x2 = coordinates.max(axis=0) + 1
    return image.convert("L").crop(
        (
            max(0, int(x1) - padding),
            max(0, int(y1) - padding),
            min(image.width, int(x2) + padding),
            min(image.height, int(y2) + padding),
        )
    )


def normalize_crop(image: Image.Image, size: tuple[int, int] = (64, 160)) -> Image.Image:
    """Tight-crop ink and letterbox it onto a fixed white canvas."""

    height, width = size
    crop = ImageOps.autocontrast(crop_to_ink(image))
    crop.thumbnail((width - 8, height - 8), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), 255)
    x = (width - crop.width) // 2
    y = (height - crop.height) // 2
    canvas.paste(crop, (x, y))
    return canvas


def to_tensor(image: Image.Image, size: tuple[int, int] = (64, 160)) -> torch.Tensor:
    """Convert to a 1xHxW float tensor where ink is positive."""

    normalized = normalize_crop(image, size)
    pixels = np.asarray(normalized, dtype=np.float32) / 255.0
    return torch.from_numpy(1.0 - pixels).unsqueeze(0)


def propose_regions(
    image: Image.Image,
    *,
    min_width: int = 14,
    min_height: int = 8,
    max_regions: int = 64,
) -> list[Box]:
    """Group connected glyphs into word-like regions without requiring OCR."""

    mask = ink_mask(image)
    # Anisotropic dilation joins glyphs inside words more readily than separate words.
    joined = ndimage.binary_dilation(mask, structure=np.ones((3, 7)), iterations=2)
    joined = ndimage.binary_closing(joined, structure=np.ones((3, 5)), iterations=1)
    labels, count = ndimage.label(joined)
    objects = ndimage.find_objects(labels)
    boxes: list[Box] = []
    for index in range(count):
        slices = objects[index]
        if slices is None:
            continue
        ys, xs = slices
        raw_width, raw_height = xs.stop - xs.start, ys.stop - ys.start
        if raw_height < min_height or raw_width < min_width:
            continue
        if raw_width * raw_height > image.width * image.height * 0.25:
            continue
        box = Box(
            max(0, xs.start - 3),
            max(0, ys.start - 3),
            min(image.width, xs.stop + 3),
            min(image.height, ys.stop + 3),
        )
        if box.width >= min_width and box.height >= min_height:
            boxes.append(box)
    boxes.sort(key=lambda box: (box.y1 // max(1, min_height), box.x1))
    return boxes[:max_regions]
