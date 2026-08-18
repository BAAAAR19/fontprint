"""Command-line entry point for the complete Fontprint workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer
from PIL import Image, ImageDraw

from fontprint import __version__
from fontprint.analyzer import Correction, FontprintAnalyzer
from fontprint.benchmark import run_benchmark
from fontprint.config import TrainConfig
from fontprint.export import export_onnx
from fontprint.fonts import discover_fonts
from fontprint.synthesis import render_document
from fontprint.training import train_model

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="markdown",
    help="Open-set typography anomaly detection for document forensics.",
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"fontprint {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Fontprint command-line interface."""


@app.command("fonts")
def list_fonts(
    roots: Annotated[
        list[Path] | None, typer.Option("--root", help="Font file or directory.")
    ] = None,
    limit: Annotated[int, typer.Option(min=1)] = 30,
) -> None:
    """List usable local font faces."""

    fonts = discover_fonts(roots, limit=limit)
    for font in fonts:
        typer.echo(f"{font.label}\t{font.path}")
    typer.echo(f"\n{len(fonts)} usable font faces")


@app.command()
def synthesize(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("docs/example-case.png"),
    roots: Annotated[list[Path] | None, typer.Option("--font-root")] = None,
    tampered: Annotated[bool, typer.Option("--tampered/--clean")] = True,
    seed: Annotated[int, typer.Option()] = 23,
    show_truth: Annotated[bool, typer.Option(help="Draw the synthetic ground-truth box.")] = False,
) -> None:
    """Create a fictional, license-safe document specimen."""

    fonts = discover_fonts(roots, limit=12)
    if len(fonts) < 2:
        raise typer.BadParameter("at least two usable local fonts are required")
    document = render_document(fonts[0].path, fonts[-1].path, tampered=tampered, seed=seed)
    image = document.image.copy()
    if show_truth:
        draw = ImageDraw.Draw(image)
        for box, changed in zip(document.boxes, document.tampered, strict=True):
            if changed:
                draw.rectangle(box.as_tuple(), outline=(220, 40, 55), width=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    typer.echo(f"Wrote {output}")


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/base.yaml"),
    font_roots: Annotated[list[Path] | None, typer.Option("--font-root")] = None,
    quick: Annotated[bool, typer.Option(help="Two-epoch smoke run on eight fonts.")] = False,
) -> None:
    """Train, evaluate, conformally calibrate, and save an inference bundle."""

    settings = TrainConfig.from_yaml(config)
    updates: dict[str, object] = {}
    if font_roots:
        updates["font_roots"] = font_roots
    if quick:
        updates.update(
            max_fonts=8,
            holdout_fonts=4,
            fonts_per_batch=4,
            samples_per_font=32,
            validation_samples_per_font=48,
            epochs=2,
        )
    settings = settings.model_copy(update=updates)

    def progress(row: dict[str, float]) -> None:
        typer.echo(
            f"epoch {int(row['epoch']):02d}/{settings.epochs} "
            f"loss={row['train_loss']:.4f} lr={row['learning_rate']:.2e}"
        )

    path, metrics = train_model(settings, progress=progress)
    typer.echo(f"\nSaved {path}")
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def analyze(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-m", exists=True)] = Path(
        "artifacts/fontprint.pt"
    ),
    overlay: Annotated[Path | None, typer.Option("--overlay", "-o")] = None,
    correction: Annotated[
        str,
        typer.Option(
            "--correction",
            help="Multiple-testing correction across regions: `bh`, `holm`, or `none`.",
        ),
    ] = "bh",
) -> None:
    """Analyze a document and print machine-readable evidence."""

    if correction not in {"bh", "holm", "none"}:
        raise typer.BadParameter("correction must be 'bh', 'holm', or 'none'")
    analyzer = FontprintAnalyzer.from_checkpoint(
        checkpoint, correction=cast(Correction, correction)
    )
    with Image.open(image) as document:
        report = analyzer.analyze(document)
        typer.echo(json.dumps(report.to_dict(), indent=2))
        if overlay is not None:
            overlay.parent.mkdir(parents=True, exist_ok=True)
            analyzer.draw_overlay(document, report).save(overlay)
            typer.echo(f"Wrote overlay to {overlay}", err=True)


@app.command()
def benchmark(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-m", exists=True)] = Path(
        "artifacts/fontprint.pt"
    ),
    font_roots: Annotated[list[Path] | None, typer.Option("--font-root")] = None,
    documents: Annotated[int, typer.Option(min=2, help="Alternating tampered/clean pages.")] = 24,
    seed: Annotated[int, typer.Option()] = 101,
    proposals: Annotated[
        bool,
        typer.Option("--proposals/--oracle-boxes", help="Score the OCR-free proposer too."),
    ] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = Path(
        "artifacts/benchmark.json"
    ),
    correction: Annotated[
        str,
        typer.Option("--correction", help="Multiple-testing correction: `bh`, `holm`, or `none`."),
    ] = "bh",
    markdown: Annotated[bool, typer.Option(help="Print a paste-ready summary table.")] = False,
) -> None:
    """Measure end-to-end detection quality on controlled synthetic substitutions."""

    if correction not in {"bh", "holm", "none"}:
        raise typer.BadParameter("correction must be 'bh', 'holm', or 'none'")
    fonts = discover_fonts(font_roots, limit=20)
    if len(fonts) < 2:
        raise typer.BadParameter("at least two usable local fonts are required")
    analyzer = FontprintAnalyzer.from_checkpoint(
        checkpoint, correction=cast(Correction, correction)
    )
    report = run_benchmark(
        analyzer,
        fonts,
        documents=documents,
        seed=seed,
        use_proposals=proposals,
    )
    typer.echo(report.to_markdown() if markdown else json.dumps(report.summary, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote {output}", err=True)


@app.command("export-onnx")
def onnx_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-m", exists=True)] = Path(
        "artifacts/fontprint.pt"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/fontprint.onnx"),
) -> None:
    """Export the encoder plus a JSON calibration sidecar."""

    typer.echo(f"Wrote {export_onnx(checkpoint, output)}")


@app.command()
def serve(
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", "-m")] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Serve the versioned FastAPI inference API."""

    try:
        import uvicorn
    except ImportError as error:
        raise typer.BadParameter("install the API extra with: pip install -e '.[api]'") from error
    from fontprint.api import create_app

    uvicorn.run(create_app(checkpoint), host=host, port=port)


@app.command()
def demo(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-m", exists=True)] = Path(
        "artifacts/fontprint.pt"
    ),
    share: Annotated[bool, typer.Option(help="Ask Gradio for a temporary public URL.")] = False,
) -> None:
    """Launch the human-in-the-loop evidence desk."""

    from fontprint.demo import create_demo

    create_demo(checkpoint).launch(share=share)


if __name__ == "__main__":
    app()
