from __future__ import annotations

from pathlib import Path

import pytest

from fontprint.fonts import FontRecord, discover_fonts


@pytest.fixture(scope="session")
def fonts() -> list[FontRecord]:
    records = discover_fonts(limit=4)
    if len(records) < 2:
        pytest.skip("test host has fewer than two Latin fonts")
    return records


@pytest.fixture()
def tmp_checkpoint(tmp_path: Path) -> Path:
    return tmp_path / "model.pt"
