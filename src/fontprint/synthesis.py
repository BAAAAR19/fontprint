"""Deterministic synthetic training samples and showcase documents."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset, Sampler

from fontprint.fonts import FontRecord
from fontprint.preprocessing import Box, to_tensor

WORDS = (
    "archive",
    "signal",
    "ledger",
    "evidence",
    "invoice",
    "receipt",
    "amount",
    "balance",
    "reference",
    "approved",
    "specimen",
    "document",
    "verified",
    "date",
    "account",
    "subtotal",
    "quantity",
    "delivery",
    "contract",
    "original",
    "memo",
    "confidential",
    "typography",
    "analysis",
    "signature",
    "payment",
    "record",
)


@dataclass(frozen=True, slots=True)
class SyntheticDocument:
    image: Image.Image
    boxes: tuple[Box, ...]
    tampered: tuple[bool, ...]
    words: tuple[str, ...]


def render_word(
    text: str,
    font_path: str | Path,
    *,
    seed: int,
    canvas_size: tuple[int, int] = (192, 72),
    augment: bool = True,
) -> Image.Image:
    """Render a word with lightweight print/scan domain randomization."""

    rng = random.Random(seed)
    width, height = canvas_size
    background = rng.randint(242, 255)
    canvas = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    size = rng.randint(31, 45) if augment else 38
    font = ImageFont.truetype(str(font_path), size=size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(2, (width - text_width) // 2 + (rng.randint(-6, 6) if augment else 0))
    y = max(1, (height - text_height) // 2 - bbox[1] + (rng.randint(-4, 4) if augment else 0))
    ink = rng.randint(0, 35)
    draw.text((x, y), text, font=font, fill=ink)

    if augment:
        angle = rng.uniform(-2.2, 2.2)
        canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=background)
        if rng.random() < 0.65:
            canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.65)))
        array = np.asarray(canvas, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0, rng.uniform(0.4, 3.0), array.shape)
        canvas = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="L")
        if rng.random() < 0.3:
            buffer = io.BytesIO()
            canvas.save(buffer, format="JPEG", quality=rng.randint(55, 88))
            buffer.seek(0)
            canvas = Image.open(buffer).convert("L")
    return canvas


class SyntheticFontDataset(Dataset[tuple[torch.Tensor, int]]):
    """Infinite-variety deterministic word crops generated from local fonts."""

    def __init__(
        self,
        fonts: list[FontRecord],
        samples_per_font: int,
        image_size: tuple[int, int] = (64, 160),
        seed: int = 17,
        augment: bool = True,
    ) -> None:
        if len(fonts) < 2:
            raise ValueError("at least two usable fonts are required")
        self.fonts = fonts
        self.samples_per_font = samples_per_font
        self.image_size = image_size
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return len(self.fonts) * self.samples_per_font

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = index // self.samples_per_font
        local_index = index % self.samples_per_font
        sample_seed = self.seed + label * 1_000_003 + local_index
        word = WORDS[sample_seed % len(WORDS)]
        image = render_word(
            word,
            self.fonts[label].path,
            seed=sample_seed,
            augment=self.augment,
        )
        return to_tensor(image, self.image_size), label


class PKBatchSampler(Sampler[list[int]]):
    """Yield P font classes x K samples, guaranteeing positive contrastive pairs."""

    def __init__(
        self,
        num_classes: int,
        samples_per_class_total: int,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int = 17,
    ) -> None:
        self.num_classes = num_classes
        self.total = samples_per_class_total
        self.p = classes_per_batch
        self.k = samples_per_class
        self.seed = seed
        self.epoch = 0
        if self.p > num_classes:
            raise ValueError("classes_per_batch cannot exceed num_classes")

    def __len__(self) -> int:
        return max(1, (self.num_classes * self.total) // (self.p * self.k))

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            classes = rng.sample(range(self.num_classes), self.p)
            yield [
                label * self.total + rng.randrange(self.total)
                for label in classes
                for _ in range(self.k)
            ]


def render_document(
    primary_font: str | Path,
    alternate_font: str | Path | None = None,
    *,
    tampered: bool = True,
    seed: int = 23,
) -> SyntheticDocument:
    """Render a fictional invoice with one controlled font substitution."""

    rng = random.Random(seed)
    page = Image.new("RGB", (960, 620), (247, 244, 236))
    draw = ImageDraw.Draw(page)
    draw.rounded_rectangle(
        (35, 30, 925, 585), radius=12, fill=(255, 254, 250), outline=(38, 47, 54), width=2
    )
    draw.text(
        (70, 58),
        "NORTHSTAR SUPPLY CO.",
        font=ImageFont.truetype(str(primary_font), 34),
        fill=(20, 28, 34),
    )
    draw.text(
        (70, 112),
        "DOCUMENT / 0427",
        font=ImageFont.truetype(str(primary_font), 19),
        fill=(73, 84, 89),
    )
    draw.line((70, 150, 890, 150), fill=(185, 184, 176), width=2)

    rows = [
        ("ARCHIVE REELS", "$184.00"),
        ("SIGNAL ADAPTER", "$72.50"),
        ("FIELD NOTEBOOK", "$28.00"),
        ("DELIVERY", "$16.00"),
        ("TOTAL DUE", "$300.50"),
    ]
    boxes: list[Box] = []
    changed: list[bool] = []
    words: list[str] = []
    y = 205
    target = 2 if tampered and alternate_font else -1
    for index, (item, amount) in enumerate(rows):
        is_changed = index == target
        face = alternate_font if is_changed else primary_font
        assert face is not None
        font = ImageFont.truetype(str(face), 27)
        draw.text((88, y), item, font=font, fill=(27, 32, 35))
        amount_bbox = draw.textbbox((0, 0), amount, font=font)
        amount_x = 855 - (amount_bbox[2] - amount_bbox[0])
        draw.text((amount_x, y), amount, font=font, fill=(27, 32, 35))
        combined = draw.textbbox((88, y), item, font=font)
        boxes.append(Box(82, int(combined[1] - 5), 870, int(combined[3] + 6)))
        changed.append(is_changed)
        words.append(f"{item} {amount}")
        y += 66

    draw.line((585, 515, 870, 515), fill=(80, 84, 82), width=2)
    draw.text(
        (70, 548),
        f"SPECIMEN • SEED {seed:04d}",
        font=ImageFont.truetype(str(primary_font), 15),
        fill=(114, 116, 111),
    )

    array = np.asarray(page, dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, 1.2, array.shape)
    page = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    if rng.random() < 0.5:
        page = page.filter(ImageFilter.GaussianBlur(0.15))
    return SyntheticDocument(page, tuple(boxes), tuple(changed), tuple(words))
