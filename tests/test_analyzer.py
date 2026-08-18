from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from fontprint.analyzer import FontprintAnalyzer
from fontprint.calibration import DistanceCalibrator
from fontprint.preprocessing import Box


class MeanInkEncoder(nn.Module):
    embedding_dim = 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ink = inputs.mean(dim=(1, 2, 3))
        vectors = torch.stack((1.0 - ink, ink * 3.0), dim=1)
        return torch.nn.functional.normalize(vectors, dim=1)


def test_analyzer_flags_a_visual_style_outlier() -> None:
    image = Image.new("L", (180, 60), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 10, 45, 40), fill=180)
    draw.rectangle((65, 10, 105, 40), fill=180)
    draw.rectangle((125, 10, 165, 40), fill=0)
    boxes = [Box(5, 10, 45, 40), Box(65, 10, 105, 40), Box(125, 10, 165, 40)]
    calibrator = DistanceCalibrator(np.linspace(0.001, 0.01, 30), alpha=0.1)
    analyzer = FontprintAnalyzer(MeanInkEncoder(), calibrator)  # type: ignore[arg-type]
    report = analyzer.analyze(image, boxes)
    assert report.review_recommended
    assert report.verdict == "typographic outlier detected"
    assert report.regions[2].is_anomaly
    assert report.to_dict()["caveat"]
    assert analyzer.draw_overlay(image, report).size == image.size


def test_analyzer_requires_three_regions() -> None:
    analyzer = FontprintAnalyzer(  # type: ignore[arg-type]
        MeanInkEncoder(), DistanceCalibrator(np.array([0.01, 0.02]))
    )
    try:
        analyzer.analyze(Image.new("L", (20, 20), 255), [Box(1, 1, 5, 5)])
    except ValueError as error:
        assert "at least three" in str(error)
    else:
        raise AssertionError("expected a ValueError")
