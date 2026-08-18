#!/usr/bin/env python3
"""Fetch a small, pinned OFL font benchmark directly from Google Fonts."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

GOOGLE_FONTS_COMMIT = "e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7"
FONT_PATHS = (
    "ofl/abrilfatface/AbrilFatface-Regular.ttf",
    "ofl/alfaslabone/AlfaSlabOne-Regular.ttf",
    "ofl/arvo/Arvo-Regular.ttf",
    "ofl/balsamiqsans/BalsamiqSans-Regular.ttf",
    "ofl/caveatbrush/CaveatBrush-Regular.ttf",
    "ofl/cormorantgaramond/CormorantGaramond[wght].ttf",
    "ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",
    "ofl/firasans/FiraSans-Regular.ttf",
    "ofl/fraunces/Fraunces[SOFT,WONK,opsz,wght].ttf",
    "ofl/inknutantiqua/InknutAntiqua-Regular.ttf",
    "ofl/jetbrainsmono/JetBrainsMono[wght].ttf",
    "ofl/lexend/Lexend[wght].ttf",
    "ofl/limelight/Limelight-Regular.ttf",
    "ofl/montserrat/Montserrat[wght].ttf",
    "ofl/notoserif/NotoSerif[wdth,wght].ttf",
    "ofl/oswald/Oswald[wght].ttf",
    "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "ofl/rubik/Rubik[wght].ttf",
    "ofl/spacemono/SpaceMono-Regular.ttf",
    "ofl/spectral/Spectral-Regular.ttf",
)


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/google/fonts/{GOOGLE_FONTS_COMMIT}/{path}"


def download(url: str, destination: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fontprint-dataset-fetcher/0.1"})
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return payload


def fetch(destination: Path) -> Path:
    records: list[dict[str, str]] = []
    for source_path in FONT_PATHS:
        family = Path(source_path).parent.name
        font_destination = destination / family / Path(source_path).name
        payload = download(raw_url(source_path), font_destination)
        license_source = f"ofl/{family}/OFL.txt"
        license_destination = destination / family / "OFL.txt"
        if not license_destination.exists():
            download(raw_url(license_source), license_destination)
        records.append(
            {
                "file": str(font_destination.relative_to(destination)),
                "source": raw_url(source_path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "license": str(license_destination.relative_to(destination)),
            }
        )
        print(f"fetched {source_path}")

    manifest = {
        "source_repository": "https://github.com/google/fonts",
        "commit": GOOGLE_FONTS_COMMIT,
        "license_note": "Each face is paired with the upstream OFL.txt file.",
        "fonts": records,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/fonts"))
    arguments = parser.parse_args()
    print(f"wrote {fetch(arguments.destination)}")


if __name__ == "__main__":
    main()
