from __future__ import annotations

from pathlib import Path

from fontprint.fonts import FontRecord, default_font_roots, discover_fonts, inspect_font


def test_default_roots_exist_on_this_platform() -> None:
    assert all(root.exists() for root in default_font_roots())


def test_inspect_font_reads_public_names(fonts) -> None:  # type: ignore[no-untyped-def]
    record = inspect_font(fonts[0].path)
    assert record is not None
    assert record.label == record.label.strip()
    assert record.path.is_absolute()


def test_inspect_font_rejects_non_font_files(tmp_path: Path) -> None:
    decoy = tmp_path / "not-a-font.ttf"
    decoy.write_bytes(b"definitely not a font")
    assert inspect_font(decoy) is None
    assert inspect_font(tmp_path / "missing.ttf") is None


def test_discovery_is_deduplicated_ordered_and_limited(fonts) -> None:  # type: ignore[no-untyped-def]
    roots = [font.path for font in fonts]
    records = discover_fonts(roots)
    labels = [record.label for record in records]
    assert len(labels) == len(set(labels))
    assert records == discover_fonts(roots), "discovery must be deterministic"
    assert len(discover_fonts(roots, limit=1)) == 1


def test_discovery_skips_directories_without_fonts(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("no fonts here", encoding="utf-8")
    assert discover_fonts([tmp_path]) == []
    assert discover_fonts([tmp_path / "missing-directory"]) == []


def test_font_record_label_joins_family_and_style() -> None:
    record = FontRecord(path=Path("/tmp/x.ttf"), family="Fira Sans", style="Regular")
    assert record.label == "Fira Sans Regular"
