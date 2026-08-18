from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient

from fontprint.api import create_app
from fontprint.calibration import DistanceCalibrator
from fontprint.checkpoint import save_checkpoint
from fontprint.index import PrototypeIndex
from fontprint.model import StyleEncoder
from fontprint.synthesis import render_document


def test_api_exposes_degraded_readiness_without_model() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is False
        assert client.get("/ready").status_code == 503
        assert client.get("/v1/model").status_code == 503
        assert client.post("/v1/analyze", files={"image": ("x.png", b"no")}).status_code == 503


def test_api_model_metadata_and_inference(tmp_checkpoint, fonts) -> None:  # type: ignore[no-untyped-def]
    encoder = StyleEncoder(embedding_dim=16)
    calibrator = DistanceCalibrator(np.linspace(0.0, 2.0, 100), alpha=0.05)
    index = PrototypeIndex(["reference"], np.ones((1, 16), dtype=np.float32))
    save_checkpoint(tmp_checkpoint, encoder, calibrator, index, image_size=(64, 160))
    document = render_document(fonts[0].path, fonts[1].path).image
    buffer = io.BytesIO()
    document.save(buffer, format="PNG")

    with TestClient(create_app(tmp_checkpoint)) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/ready").json()["status"] == "ready"
        model = client.get("/v1/model")
        assert model.status_code == 200
        assert model.json()["embedding_dim"] == 16
        response = client.post(
            "/v1/analyze?include_overlay=true",
            files={"image": ("specimen.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        assert len(response.json()["regions"]) >= 3
        assert response.json()["overlay_png_base64"]


def test_api_rejects_bad_images(tmp_checkpoint) -> None:  # type: ignore[no-untyped-def]
    encoder = StyleEncoder(embedding_dim=16)
    calibrator = DistanceCalibrator(np.array([0.1, 0.2]))
    index = PrototypeIndex(["reference"], np.ones((1, 16), dtype=np.float32))
    save_checkpoint(tmp_checkpoint, encoder, calibrator, index, image_size=(64, 160))
    with TestClient(create_app(tmp_checkpoint)) as client:
        response = client.post("/v1/analyze", files={"image": ("bad.png", b"not-an-image")})
        assert response.status_code == 415
