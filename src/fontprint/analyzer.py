"""Document-level typography consistency analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fontprint.calibration import DistanceCalibrator
from fontprint.checkpoint import load_checkpoint
from fontprint.index import PrototypeIndex
from fontprint.model import StyleEncoder
from fontprint.preprocessing import Box, propose_regions, to_tensor


@dataclass(frozen=True, slots=True)
class RegionEvidence:
    region_id: int
    box: tuple[int, int, int, int]
    anomaly_score: float
    p_value: float
    is_anomaly: bool
    nearest_style: str | None
    style_similarity: float | None


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    verdict: str
    review_recommended: bool
    reference_region_id: int
    threshold: float
    regions: tuple[RegionEvidence, ...]
    caveat: str = (
        "Fontprint measures typographic inconsistency; it does not establish that a document "
        "is authentic or fraudulent."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "review_recommended": self.review_recommended,
            "reference_region_id": self.reference_region_id,
            "threshold": self.threshold,
            "regions": [asdict(region) for region in self.regions],
            "caveat": self.caveat,
        }


class FontprintAnalyzer:
    """Compare regions to a document's medoid style and return calibrated evidence."""

    def __init__(
        self,
        encoder: StyleEncoder,
        calibrator: DistanceCalibrator,
        index: PrototypeIndex | None = None,
        *,
        image_size: tuple[int, int] = (64, 160),
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.calibrator = calibrator
        self.index = index
        self.image_size = image_size

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> FontprintAnalyzer:
        encoder, calibrator, index, metadata = load_checkpoint(path, device)
        return cls(
            encoder,
            calibrator,
            index,
            image_size=metadata["image_size"],
            device=device,
        )

    @torch.inference_mode()
    def embed_regions(self, image: Image.Image, boxes: Sequence[Box]) -> np.ndarray:
        if not boxes:
            raise ValueError("no typographic regions were supplied or detected")
        tensors = [to_tensor(image.crop(box.as_tuple()), self.image_size) for box in boxes]
        batch = torch.stack(tensors).to(self.device)
        return np.asarray(self.encoder(batch).cpu().numpy(), dtype=np.float32)

    def analyze(
        self,
        image: Image.Image,
        boxes: Sequence[Box] | None = None,
    ) -> AnalysisReport:
        regions = list(boxes) if boxes is not None else propose_regions(image)
        if len(regions) < 3:
            raise ValueError(
                "at least three text regions are required to infer the document's dominant style"
            )
        embeddings = self.embed_regions(image.convert("RGB"), regions)
        pairwise = np.clip(1.0 - embeddings @ embeddings.T, 0.0, 2.0)
        # A median-distance medoid is robust to a minority of substituted regions.
        medoid_index = int(np.argmin(np.median(pairwise, axis=1)))
        scores = pairwise[:, medoid_index]

        evidence: list[RegionEvidence] = []
        for region_id, (box, score, embedding) in enumerate(
            zip(regions, scores, embeddings, strict=True)
        ):
            matches = self.index.search(embedding, k=1) if self.index is not None else []
            evidence.append(
                RegionEvidence(
                    region_id=region_id,
                    box=box.as_tuple(),
                    anomaly_score=float(score),
                    p_value=self.calibrator.p_value(float(score)),
                    is_anomaly=region_id != medoid_index
                    and self.calibrator.is_anomaly(float(score)),
                    nearest_style=matches[0].label if matches else None,
                    style_similarity=matches[0].similarity if matches else None,
                )
            )

        review = any(item.is_anomaly for item in evidence)
        return AnalysisReport(
            verdict="typographic outlier detected" if review else "no calibrated outlier detected",
            review_recommended=review,
            reference_region_id=medoid_index,
            threshold=self.calibrator.threshold,
            regions=tuple(evidence),
        )

    @staticmethod
    def draw_overlay(image: Image.Image, report: AnalysisReport) -> Image.Image:
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas, "RGBA")
        font = ImageFont.load_default()
        for region in report.regions:
            color = (216, 62, 66, 255) if region.is_anomaly else (31, 139, 110, 230)
            fill = (216, 62, 66, 35) if region.is_anomaly else (31, 139, 110, 22)
            draw.rectangle(region.box, fill=fill, outline=color, width=3)
            label = f"#{region.region_id} d={region.anomaly_score:.3f} p={region.p_value:.3f}"
            x1, y1, _, _ = region.box
            label_box = draw.textbbox((x1, max(0, y1 - 16)), label, font=font)
            draw.rectangle(label_box, fill=(17, 24, 28, 220))
            draw.text((x1, max(0, y1 - 16)), label, font=font, fill=(255, 255, 255, 255))
        return canvas
