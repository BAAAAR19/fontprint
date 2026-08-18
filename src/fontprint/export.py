"""Deployment exports for runtimes that do not embed Python or PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fontprint.checkpoint import load_checkpoint


def export_onnx(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    opset: int = 18,
) -> Path:
    encoder, calibrator, index, metadata = load_checkpoint(checkpoint)
    height, width = metadata["image_size"]
    dummy = torch.zeros(1, 1, height, width)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder,
        (dummy,),
        output,
        input_names=["word_crop"],
        output_names=["style_embedding"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=opset,
        dynamo=True,
    )
    sidecar = {
        "image_size": list(metadata["image_size"]),
        "calibration": calibrator.to_dict(),
        "index": index.to_dict(),
        "metrics": metadata["metrics"],
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return output
