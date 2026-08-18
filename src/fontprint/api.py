"""FastAPI transport for Fontprint inference."""

from __future__ import annotations

import base64
import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from fontprint import __version__
from fontprint.analyzer import FontprintAnalyzer

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def create_app(checkpoint: str | Path | None = None) -> FastAPI:
    checkpoint_path = Path(checkpoint) if checkpoint else None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.analyzer = (
            FontprintAnalyzer.from_checkpoint(checkpoint_path)
            if checkpoint_path is not None and checkpoint_path.exists()
            else None
        )
        yield

    api = FastAPI(
        title="Fontprint API",
        version=__version__,
        description="Calibrated typographic inconsistency evidence for document review.",
        lifespan=lifespan,
    )

    @api.get("/health", tags=["operations"])
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": api.state.analyzer is not None,
            "version": __version__,
        }

    @api.get("/ready", tags=["operations"])
    async def readiness() -> dict[str, str]:
        if api.state.analyzer is None:
            raise HTTPException(status_code=503, detail="model checkpoint is not loaded")
        return {"status": "ready"}

    @api.get("/v1/model", tags=["operations"])
    async def model_info() -> dict[str, Any]:
        analyzer: FontprintAnalyzer | None = api.state.analyzer
        if analyzer is None:
            raise HTTPException(status_code=503, detail="model checkpoint is not loaded")
        return {
            "embedding_dim": analyzer.encoder.embedding_dim,
            "image_size": analyzer.image_size,
            "calibration_alpha": analyzer.calibrator.alpha,
            "distance_threshold": analyzer.calibrator.threshold,
            "indexed_styles": len(analyzer.index.labels) if analyzer.index else 0,
        }

    @api.post("/v1/analyze", tags=["inference"])
    async def analyze(
        image: Annotated[UploadFile, File(description="PNG, JPEG, or WebP document image")],
        include_overlay: Annotated[bool, Query()] = False,
    ) -> dict[str, object]:
        analyzer: FontprintAnalyzer | None = api.state.analyzer
        if analyzer is None:
            raise HTTPException(status_code=503, detail="model checkpoint is not loaded")
        payload = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image exceeds the 10 MB limit")
        try:
            document = Image.open(io.BytesIO(payload))
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(status_code=415, detail="unsupported or corrupt image") from error
        if document.width * document.height > 25_000_000:
            raise HTTPException(status_code=413, detail="image exceeds the 25 megapixel limit")
        try:
            document.load()
        except OSError as error:
            raise HTTPException(status_code=415, detail="unsupported or corrupt image") from error
        try:
            report = analyzer.analyze(document)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        response = report.to_dict()
        if include_overlay:
            overlay = analyzer.draw_overlay(document, report)
            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            response["overlay_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return response

    return api


app = create_app(os.getenv("FONTPRINT_CHECKPOINT"))
