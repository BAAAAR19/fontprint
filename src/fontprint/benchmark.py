"""End-to-end detection benchmark on controlled synthetic substitutions.

Embedding metrics (retrieval accuracy, pair AUROC) describe the style space, not the
product. This module answers the operational question instead: given a document, how
often does the full pipeline flag a substituted line, how often does it cry wolf on a
clean page, and does the conformal alpha hold empirically at document scale?
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from fontprint.analyzer import AnalysisReport, FontprintAnalyzer
from fontprint.fonts import FontRecord
from fontprint.metrics import precision_recall_f1, roc_auc
from fontprint.preprocessing import Box, propose_regions
from fontprint.synthesis import render_document


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    """One benchmark document and what the analyzer concluded about it."""

    seed: int
    primary_font: str
    alternate_font: str
    tampered: bool
    flagged: bool
    localized: bool
    num_regions: int
    max_anomaly_score: float
    min_p_value: float

    @property
    def true_positive(self) -> bool:
        return self.tampered and self.localized

    @property
    def false_positive(self) -> bool:
        return not self.tampered and self.flagged


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate detection quality plus the per-document rows behind it."""

    summary: dict[str, float]
    documents: tuple[DocumentOutcome, ...] = field(repr=False)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "settings": self.settings,
            "documents": [asdict(document) for document in self.documents],
        }

    def to_markdown(self) -> str:
        """Render the summary as a table that can be pasted into a README or PR."""

        rows = "\n".join(
            f"| `{name}` | {value:.4g} |" for name, value in sorted(self.summary.items())
        )
        return f"| metric | value |\n|---|---|\n{rows}\n"


def _containment(inner: Box, outer: tuple[int, int, int, int]) -> float:
    """Fraction of `inner` that falls inside `outer`; robust to size mismatch."""

    x1 = max(inner.x1, outer[0])
    y1 = max(inner.y1, outer[1])
    x2 = min(inner.x2, outer[2])
    y2 = min(inner.y2, outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / max(inner.area, 1)


def _region_labels(
    regions: Sequence[Box],
    truth_boxes: Sequence[Box],
    tampered_flags: Sequence[bool],
    *,
    overlap: float,
) -> list[bool]:
    """Label each analyzed region by overlap with the substituted ground-truth line."""

    substituted = [box for box, changed in zip(truth_boxes, tampered_flags, strict=True) if changed]
    return [
        any(_containment(region, truth.as_tuple()) >= overlap for truth in substituted)
        for region in regions
    ]


def _document_row(
    report: AnalysisReport,
    labels: Sequence[bool],
    *,
    seed: int,
    primary: str,
    alternate: str,
    tampered: bool,
) -> DocumentOutcome:
    flagged = [region.is_anomaly for region in report.regions]
    scores = [region.anomaly_score for region in report.regions]
    p_values = [region.p_value for region in report.regions]
    localized = any(hit and truth for hit, truth in zip(flagged, labels, strict=True))
    return DocumentOutcome(
        seed=seed,
        primary_font=primary,
        alternate_font=alternate,
        tampered=tampered,
        flagged=any(flagged),
        localized=localized,
        num_regions=len(report.regions),
        max_anomaly_score=float(max(scores, default=0.0)),
        min_p_value=float(min(p_values, default=1.0)),
    )


def run_benchmark(
    analyzer: FontprintAnalyzer,
    fonts: Sequence[FontRecord],
    *,
    documents: int = 24,
    seed: int = 101,
    use_proposals: bool = False,
    overlap: float = 0.5,
) -> BenchmarkReport:
    """Score the analyzer on alternating tampered/clean fictional documents.

    Half of the documents contain exactly one substituted line, half are internally
    consistent. With ``use_proposals`` the OCR-free region proposer runs too, so the
    numbers measure the deployed pipeline rather than the encoder in isolation.
    """

    if len(fonts) < 2:
        raise ValueError("benchmarking needs at least two distinct fonts")
    if documents < 2:
        raise ValueError("benchmarking needs at least two documents")

    rng = random.Random(seed)
    outcomes: list[DocumentOutcome] = []
    region_scores: list[float] = []
    region_truth: list[bool] = []
    region_flags: list[bool] = []
    coverage: list[float] = []
    skipped = 0

    for step in range(documents):
        primary, alternate = rng.sample(range(len(fonts)), 2)
        tampered = step % 2 == 0
        document_seed = seed + step * 7
        document = render_document(
            fonts[primary].path,
            fonts[alternate].path,
            tampered=tampered,
            seed=document_seed,
        )
        if use_proposals:
            regions = propose_regions(document.image)
            recovered = [
                any(_containment(region, truth.as_tuple()) >= overlap for region in regions)
                for truth in document.boxes
            ]
            coverage.append(float(np.mean(recovered)) if recovered else 0.0)
        else:
            regions = list(document.boxes)
            coverage.append(1.0)
        if len(regions) < 3:
            skipped += 1
            continue

        report = analyzer.analyze(document.image, regions)
        labels = _region_labels(regions, document.boxes, document.tampered, overlap=overlap)
        outcomes.append(
            _document_row(
                report,
                labels,
                seed=document_seed,
                primary=fonts[primary].label,
                alternate=fonts[alternate].label if tampered else fonts[primary].label,
                tampered=tampered,
            )
        )
        region_scores.extend(region.anomaly_score for region in report.regions)
        region_flags.extend(region.is_anomaly for region in report.regions)
        region_truth.extend(labels)

    if not outcomes:
        raise RuntimeError("no benchmark document produced enough regions to analyze")

    tampered_docs = [row for row in outcomes if row.tampered]
    clean_docs = [row for row in outcomes if not row.tampered]
    precision, recall, f1 = precision_recall_f1(np.asarray(region_flags), np.asarray(region_truth))
    consistent = ~np.asarray(region_truth, dtype=bool)
    summary = {
        "documents_scored": float(len(outcomes)),
        "documents_skipped": float(skipped),
        "document_recall": _rate(row.localized for row in tampered_docs),
        "document_flag_rate": _rate(row.flagged for row in tampered_docs),
        "document_false_positive_rate": _rate(row.flagged for row in clean_docs),
        "region_precision": precision,
        "region_recall": recall,
        "region_f1": f1,
        "region_auroc": roc_auc(np.asarray(region_scores), np.asarray(region_truth)),
        # The conformal alpha promises this stays near the nominal level on style-consistent text.
        "consistent_region_flag_rate": float(
            np.asarray(region_flags, dtype=bool)[consistent].mean() if consistent.any() else 0.0
        ),
        "nominal_alpha": float(analyzer.calibrator.alpha),
        "mean_substituted_distance": _mean(np.asarray(region_scores), np.asarray(region_truth)),
        "mean_consistent_distance": _mean(np.asarray(region_scores), consistent),
        "proposal_coverage": float(np.mean(coverage)) if coverage else 0.0,
    }
    settings = {
        "documents": documents,
        "seed": seed,
        "use_proposals": use_proposals,
        "correction": analyzer.correction,
        "overlap": overlap,
        "fonts": [font.label for font in fonts],
    }
    return BenchmarkReport(summary=summary, documents=tuple(outcomes), settings=settings)


def _rate(flags: Any) -> float:
    values = list(flags)
    return float(np.mean(values)) if values else 0.0


def _mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else 0.0
