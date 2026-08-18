"""Gradio review desk for local, human-in-the-loop exploration."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fontprint.analyzer import FontprintAnalyzer
from fontprint.fonts import discover_fonts
from fontprint.synthesis import render_document


def create_demo(checkpoint: str | Path):  # type: ignore[no-untyped-def]
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("install the demo extra with: pip install -e '.[demo]'") from error

    analyzer = FontprintAnalyzer.from_checkpoint(checkpoint)

    def inspect(image: Image.Image | None) -> tuple[Image.Image | None, str]:
        if image is None:
            return None, json.dumps({"error": "Choose or generate a document."}, indent=2)
        try:
            report = analyzer.analyze(image)
        except ValueError as error:
            return image, json.dumps({"error": str(error)}, indent=2)
        return analyzer.draw_overlay(image, report), json.dumps(report.to_dict(), indent=2)

    def specimen() -> Image.Image:
        pinned_root = Path("data/fonts")
        fonts = discover_fonts([pinned_root] if pinned_root.exists() else None, limit=10)
        if len(fonts) < 2:
            raise RuntimeError("two local Latin fonts are needed to render a specimen")
        return render_document(fonts[0].path, fonts[-1].path, tampered=True).image

    with gr.Blocks(title="Fontprint — typography evidence desk") as interface:
        gr.Markdown(
            "# Fontprint\n"
            "**Typography evidence, not an authenticity verdict.** Upload a document or generate "
            "a controlled specimen; red regions are calibrated style outliers."
        )
        with gr.Row():
            source = gr.Image(type="pil", label="Document")
            overlay = gr.Image(type="pil", label="Evidence overlay", interactive=False)
        with gr.Row():
            generate = gr.Button("Generate tampered specimen")
            analyze_button = gr.Button("Analyze typography", variant="primary")
        report = gr.Code(label="Evidence JSON", language="json")
        generate.click(specimen, outputs=source)
        analyze_button.click(inspect, inputs=source, outputs=[overlay, report])
    return interface
