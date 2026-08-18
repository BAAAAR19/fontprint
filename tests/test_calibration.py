from __future__ import annotations

import numpy as np

from fontprint.calibration import DistanceCalibrator
from fontprint.index import PrototypeIndex


def test_calibrator_round_trip_and_p_values() -> None:
    calibrator = DistanceCalibrator(np.linspace(0.01, 0.20, 20), alpha=0.1)
    restored = DistanceCalibrator.from_dict(calibrator.to_dict())
    assert restored.threshold == calibrator.threshold
    assert restored.p_value(0.5) < restored.p_value(0.05)
    assert restored.is_anomaly(0.5)


def test_fit_calibration_and_prototype_search() -> None:
    embeddings = np.array(
        [
            [1.0, 0.00],
            [1.0, 0.03],
            [1.0, 0.06],
            [1.0, 0.09],
            [0.00, 1.0],
            [0.03, 1.0],
            [0.06, 1.0],
            [0.09, 1.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    calibrator = DistanceCalibrator.fit(embeddings, labels, alpha=0.2)
    index = PrototypeIndex.fit(embeddings, labels, ["serif", "sans"])
    assert calibrator.threshold < 0.01
    assert index.search(np.array([1.0, 0.0]), k=1)[0].label == "serif"
    restored = PrototypeIndex.from_dict(index.to_dict())
    assert restored.search(np.array([0.0, 1.0]), k=1)[0].label == "sans"
