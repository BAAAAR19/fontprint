from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fontprint.analyzer import FontprintAnalyzer
from fontprint.benchmark import _containment, run_benchmark
from fontprint.calibration import DistanceCalibrator
from fontprint.preprocessing import Box


class InkStyleEncoder(nn.Module):
    """Deterministic stand-in whose embedding tracks stroke weight, not identity."""

    embedding_dim = 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ink = inputs.mean(dim=(1, 2, 3))
        vectors = torch.stack((torch.ones_like(ink), ink * 12.0), dim=1)
        return torch.nn.functional.normalize(vectors, dim=1)


def _analyzer(alpha: float = 0.1) -> FontprintAnalyzer:
    calibrator = DistanceCalibrator(np.linspace(0.0005, 0.004, 40), alpha=alpha)
    return FontprintAnalyzer(InkStyleEncoder(), calibrator)  # type: ignore[arg-type]


def test_containment_is_a_fraction_of_the_inner_box() -> None:
    assert _containment(Box(0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _containment(Box(0, 0, 10, 10), (5, 0, 15, 10)) == 0.5
    assert _containment(Box(0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_benchmark_reports_calibrated_detection_summary(fonts) -> None:  # type: ignore[no-untyped-def]
    report = run_benchmark(_analyzer(), fonts, documents=4, seed=11)
    summary = report.summary

    assert summary["documents_scored"] == 4.0
    assert len(report.documents) == 4
    assert sum(row.tampered for row in report.documents) == 2
    for name in ("document_recall", "document_false_positive_rate", "region_auroc"):
        assert 0.0 <= summary[name] <= 1.0
    assert summary["nominal_alpha"] == pytest.approx(0.1)
    assert summary["proposal_coverage"] == 1.0
    assert report.settings["use_proposals"] is False
    assert "| metric | value |" in report.to_markdown()
    assert report.to_dict()["documents"][0]["seed"] == 11


def test_benchmark_scores_the_region_proposer_end_to_end(fonts) -> None:  # type: ignore[no-untyped-def]
    report = run_benchmark(_analyzer(), fonts, documents=2, seed=5, use_proposals=True)
    assert report.settings["use_proposals"] is True
    # Proposals are found without OCR, so coverage is informative rather than perfect.
    assert 0.0 <= report.summary["proposal_coverage"] <= 1.0
    assert report.summary["documents_scored"] + report.summary["documents_skipped"] == 2.0


def test_benchmark_validates_its_inputs(fonts) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="two distinct fonts"):
        run_benchmark(_analyzer(), fonts[:1], documents=4)
    with pytest.raises(ValueError, match="two documents"):
        run_benchmark(_analyzer(), fonts, documents=1)
