from __future__ import annotations

import numpy as np
from PIL import Image

from fontprint.preprocessing import Box, normalize_crop, otsu_threshold, propose_regions, to_tensor
from fontprint.synthesis import render_document, render_word


def test_otsu_separates_bimodal_image() -> None:
    pixels = np.concatenate([np.zeros(100, dtype=np.uint8), np.full(100, 240, dtype=np.uint8)])
    threshold = otsu_threshold(pixels.reshape(10, 20))
    assert 0 <= threshold < 240


def test_word_normalization_has_expected_shape(fonts) -> None:  # type: ignore[no-untyped-def]
    image = render_word("evidence", fonts[0].path, seed=4, augment=False)
    normalized = normalize_crop(image)
    tensor = to_tensor(image)
    assert normalized.size == (160, 64)
    assert tensor.shape == (1, 64, 160)
    assert 0.0 < float(tensor.mean()) < 0.5


def test_region_proposal_ignores_page_border(fonts) -> None:  # type: ignore[no-untyped-def]
    document = render_document(fonts[0].path, fonts[1].path, tampered=True)
    boxes = propose_regions(document.image)
    assert len(boxes) >= 8
    assert all(box.area < document.image.width * document.image.height * 0.25 for box in boxes)
    assert boxes == sorted(boxes, key=lambda box: (box.y1 // 8, box.x1))


def test_region_proposal_drops_letterbox_wide_strips(fonts) -> None:  # type: ignore[no-untyped-def]
    document = render_document(fonts[0].path, fonts[1].path, tampered=True)
    default_boxes = propose_regions(document.image)
    assert all(box.width <= 8.0 * box.height for box in default_boxes)
    # Relaxing the cap can only admit more regions, never fewer.
    assert len(propose_regions(document.image, max_aspect=40.0)) >= len(default_boxes)


def test_blank_crop_is_supported() -> None:
    image = Image.new("L", (40, 20), 255)
    assert normalize_crop(image).size == (160, 64)
    assert Box(1, 2, 5, 8).as_tuple() == (1, 2, 5, 8)
