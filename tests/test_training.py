from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from fontprint.checkpoint import load_checkpoint
from fontprint.config import TrainConfig
from fontprint.training import train_model, verification_metrics


def test_train_config_derived_values_and_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "config.yaml"
    config_path.write_text("max_fonts: 3\nfonts_per_batch: 2\nsamples_per_class: 3\n")
    config = TrainConfig.from_yaml(config_path)
    assert config.batch_size == 6
    assert config.image_size == (64, 160)
    with pytest.raises(ValidationError, match="fonts_per_batch"):
        TrainConfig(max_fonts=2, fonts_per_batch=3)


def test_verification_metrics_separate_styles() -> None:
    vectors = np.array([[1.0, 0.0], [0.99, 0.05], [0.0, 1.0], [0.05, 0.99]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metrics = verification_metrics(vectors, np.array([0, 0, 1, 1]))
    assert metrics["heldout_pair_auroc"] == 1.0
    assert metrics["heldout_same_style_distance"] < metrics["heldout_different_style_distance"]


def test_tiny_training_run_produces_complete_bundle(tmp_path, fonts) -> None:  # type: ignore[no-untyped-def]
    # Four explicit font files keep discovery deterministic across developer machines and CI.
    if len(fonts) < 4:
        pytest.skip("training smoke test needs four font faces")
    config = TrainConfig(
        seed=3,
        font_roots=[font.path for font in fonts],
        max_fonts=2,
        holdout_fonts=2,
        samples_per_font=4,
        validation_samples_per_font=4,
        fonts_per_batch=2,
        samples_per_class=2,
        epochs=1,
        embedding_dim=16,
        output_dir=tmp_path,
    )
    progress: list[dict[str, float]] = []
    checkpoint, metrics = train_model(config, device=torch.device("cpu"), progress=progress.append)
    assert checkpoint.exists()
    assert len(progress) == 1
    assert 0.0 <= metrics["prototype_accuracy"] <= 1.0
    assert 0.0 <= metrics["heldout_pair_auroc"] <= 1.0
    _, calibrator, index, metadata = load_checkpoint(checkpoint)
    assert len(index.labels) == 2
    assert calibrator.scores.size == 2
    assert metadata["metrics"]["num_fonts"] == 2.0
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["device"] == "cpu"
    assert len(report["holdout_fonts"]) == 2
