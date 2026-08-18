"""Font discovery without bundling or redistributing font files."""

from __future__ import annotations

import platform
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont


@dataclass(frozen=True, slots=True)
class FontRecord:
    path: Path
    family: str
    style: str

    @property
    def label(self) -> str:
        return f"{self.family} {self.style}".strip()


def default_font_roots() -> list[Path]:
    """Return conventional font directories for the current OS."""

    system = platform.system()
    if system == "Darwin":
        roots = [Path("/System/Library/Fonts"), Path("/Library/Fonts")]
    elif system == "Windows":
        roots = [Path("C:/Windows/Fonts")]
    else:
        roots = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    roots.append(Path.home() / ".fonts")
    return [root for root in roots if root.exists()]


def _name(font: TTFont, name_id: int, fallback: str) -> str:
    table = font["name"]
    value = table.getBestFamilyName() if name_id == 1 else table.getBestSubFamilyName()
    return str(value or fallback).strip()


def inspect_font(path: str | Path) -> FontRecord | None:
    """Read a font's public names and reject files that cannot render Latin text."""

    font_path = Path(path)
    try:
        pil_font = ImageFont.truetype(str(font_path), size=32)
        if pil_font.getbbox("Evidence 2048") is None:
            return None
        with TTFont(font_path, lazy=True, fontNumber=0) as font:
            cmap = font.getBestCmap() or {}
            if not all(ord(character) in cmap for character in "AaEe09"):
                return None
            family = _name(font, 1, font_path.stem)
            style = _name(font, 2, "Regular")
    except (OSError, KeyError, ValueError):
        return None
    excluded = ("symbol", "lastresort", "keyboard", "braille", "camera")
    if family.startswith(".") or any(token in family.casefold() for token in excluded):
        return None
    return FontRecord(path=font_path.resolve(), family=family, style=style)


def discover_fonts(
    roots: Iterable[str | Path] | None = None,
    *,
    limit: int | None = None,
) -> list[FontRecord]:
    """Discover renderable TTF/OTF faces, deterministically and without duplicates."""

    search_roots = [Path(root) for root in roots] if roots else default_font_roots()
    paths: list[Path] = []
    for root in search_roots:
        if root.is_file() and root.suffix.lower() in {".ttf", ".otf"}:
            paths.append(root)
        elif root.is_dir():
            paths.extend(root.rglob("*.ttf"))
            paths.extend(root.rglob("*.otf"))

    records: list[FontRecord] = []
    seen: set[str] = set()
    for path in sorted(set(paths), key=lambda value: str(value).lower()):
        record = inspect_font(path)
        if record is None:
            continue
        key = record.label.casefold()
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records
