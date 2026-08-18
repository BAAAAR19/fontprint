from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime as ort

from fontprint.calibration import DistanceCalibrator
from fontprint.checkpoint import save_checkpoint
from fontprint.export import export_onnx
from fontprint.index import PrototypeIndex
from fontprint.model import StyleEncoder


def test_onnx_export_has_dynamic_batch_and_sidecar(tmp_checkpoint, tmp_path) -> None:  # type: ignore[no-untyped-def]
    encoder = StyleEncoder(embedding_dim=16)
    calibrator = DistanceCalibrator(np.array([0.01, 0.02]))
    index = PrototypeIndex(["reference"], np.ones((1, 16), dtype=np.float32))
    save_checkpoint(tmp_checkpoint, encoder, calibrator, index, image_size=(64, 160))
    destination = export_onnx(tmp_checkpoint, tmp_path / "model.onnx")

    onnx.checker.check_model(onnx.load(destination))
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    result = session.run(None, {"word_crop": np.zeros((2, 1, 64, 160), dtype=np.float32)})[0]
    assert result.shape == (2, 16)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)
    sidecar = json.loads(destination.with_suffix(".json").read_text())
    assert sidecar["image_size"] == [64, 160]
