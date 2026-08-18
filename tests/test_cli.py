from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from fontprint import __version__
from fontprint.calibration import DistanceCalibrator
from fontprint.checkpoint import save_checkpoint
from fontprint.cli import app
from fontprint.index import PrototypeIndex
from fontprint.model import StyleEncoder

runner = CliRunner()


def _checkpoint(path: Path) -> Path:
    encoder = StyleEncoder(embedding_dim=16)
    calibrator = DistanceCalibrator(np.linspace(0.01, 0.9, 50), alpha=0.1)
    index = PrototypeIndex(["reference"], np.ones((1, 16), dtype=np.float32))
    save_checkpoint(path, encoder, calibrator, index, image_size=(64, 160))
    return path


def test_version_flag_short_circuits() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_fonts_command_lists_discovered_faces(fonts) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["fonts", "--root", str(fonts[0].path.parent), "--limit", "3"])
    assert result.exit_code == 0
    assert "usable font faces" in result.stdout


def test_synthesize_writes_a_specimen(tmp_path: Path) -> None:
    output = tmp_path / "specimen.png"
    result = runner.invoke(app, ["synthesize", "--output", str(output), "--show-truth"])
    assert result.exit_code == 0, result.stdout
    assert output.exists()


def test_analyze_emits_evidence_json_and_overlay(tmp_path: Path) -> None:
    specimen = tmp_path / "specimen.png"
    assert runner.invoke(app, ["synthesize", "--output", str(specimen)]).exit_code == 0
    overlay = tmp_path / "evidence.png"
    result = runner.invoke(
        app,
        [
            "analyze",
            str(specimen),
            "--checkpoint",
            str(_checkpoint(tmp_path / "model.pt")),
            "--overlay",
            str(overlay),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["caveat"]
    assert len(payload["regions"]) >= 3
    assert overlay.exists()


def test_benchmark_command_writes_a_report(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--checkpoint",
            str(_checkpoint(tmp_path / "model.pt")),
            "--documents",
            "2",
            "--output",
            str(tmp_path / "benchmark.json"),
            "--markdown",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "| metric | value |" in result.stdout
    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert report["summary"]["documents_scored"] == 2.0
    assert report["settings"]["documents"] == 2


def test_analyze_rejects_a_missing_checkpoint(tmp_path: Path) -> None:
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope.png")])
    assert result.exit_code != 0
