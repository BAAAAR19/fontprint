from __future__ import annotations

import torch

from fontprint.synthesis import PKBatchSampler, SyntheticFontDataset, render_document


def test_synthetic_dataset_is_deterministic(fonts) -> None:  # type: ignore[no-untyped-def]
    dataset = SyntheticFontDataset(fonts[:2], samples_per_font=3, seed=12)
    first, first_label = dataset[1]
    second, second_label = dataset[1]
    assert first_label == second_label == 0
    assert torch.equal(first, second)
    assert len(dataset) == 6


def test_pk_sampler_has_positive_pairs() -> None:
    sampler = PKBatchSampler(4, 10, classes_per_batch=3, samples_per_class=2, seed=1)
    batch = next(iter(sampler))
    labels = [index // 10 for index in batch]
    assert len(batch) == 6
    assert sorted(labels.count(label) for label in set(labels)) == [2, 2, 2]


def test_render_document_marks_one_controlled_substitution(fonts) -> None:  # type: ignore[no-untyped-def]
    document = render_document(fonts[0].path, fonts[1].path, tampered=True, seed=9)
    assert document.image.size == (960, 620)
    assert sum(document.tampered) == 1
    assert len(document.boxes) == len(document.words) == 5
