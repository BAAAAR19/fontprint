from __future__ import annotations

import numpy as np

from fontprint.calibration import DistanceCalibrator
from fontprint.checkpoint import load_checkpoint, save_checkpoint
from fontprint.index import PrototypeIndex
from fontprint.model import StyleEncoder


def test_checkpoint_round_trip(tmp_checkpoint) -> None:  # type: ignore[no-untyped-def]
    encoder = StyleEncoder(embedding_dim=16)
    calibrator = DistanceCalibrator(np.array([0.01, 0.02, 0.03]))
    index = PrototypeIndex(["one", "two"], np.eye(2, 16, dtype=np.float32))
    save_checkpoint(
        tmp_checkpoint,
        encoder,
        calibrator,
        index,
        image_size=(64, 160),
        metrics={"accuracy": 0.8},
    )
    loaded_encoder, loaded_calibrator, loaded_index, metadata = load_checkpoint(tmp_checkpoint)
    assert loaded_encoder.embedding_dim == 16
    assert loaded_calibrator.threshold == calibrator.threshold
    assert loaded_index.labels == ["one", "two"]
    assert metadata["metrics"]["accuracy"] == 0.8
