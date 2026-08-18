"""Portable checkpoint I/O with explicit metadata and schema versioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fontprint.calibration import DistanceCalibrator
from fontprint.index import PrototypeIndex
from fontprint.model import StyleEncoder

SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    encoder: StyleEncoder,
    calibrator: DistanceCalibrator,
    index: PrototypeIndex,
    *,
    image_size: tuple[int, int],
    metrics: dict[str, float] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "embedding_dim": encoder.embedding_dim,
            "image_size": list(image_size),
            "state_dict": encoder.state_dict(),
            "calibration": calibrator.to_dict(),
            "index": index.to_dict(),
            "metrics": metrics or {},
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[StyleEncoder, DistanceCalibrator, PrototypeIndex, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    encoder = StyleEncoder(int(payload["embedding_dim"]))
    encoder.load_state_dict(payload["state_dict"])
    encoder.to(device).eval()
    calibrator = DistanceCalibrator.from_dict(payload["calibration"])
    index = PrototypeIndex.from_dict(payload["index"])
    metadata = {
        "image_size": tuple(int(value) for value in payload["image_size"]),
        "metrics": payload.get("metrics", {}),
        "schema_version": payload["schema_version"],
    }
    return encoder, calibrator, index, metadata
