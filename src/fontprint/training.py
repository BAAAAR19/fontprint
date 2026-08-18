"""Reproducible training and evaluation pipeline."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fontprint.analyzer import FontprintAnalyzer
from fontprint.calibration import DistanceCalibrator
from fontprint.checkpoint import save_checkpoint
from fontprint.config import TrainConfig
from fontprint.fonts import FontRecord, discover_fonts
from fontprint.index import PrototypeIndex
from fontprint.losses import SupervisedContrastiveLoss
from fontprint.metrics import roc_auc
from fontprint.model import StyleEncoder, TrainingModel
from fontprint.preprocessing import propose_regions
from fontprint.synthesis import PKBatchSampler, SyntheticFontDataset, render_document

ProgressCallback = Callable[[dict[str, float]], None]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def collect_embeddings(
    model: TrainingModel,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        embeddings = model.encoder(images.to(device))
        vectors.append(embeddings.cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(vectors), np.concatenate(labels)


def retrieval_metrics(
    embeddings: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], PrototypeIndex]:
    """Compute closed-set prototype accuracy and separation diagnostics."""

    index = PrototypeIndex.fit(embeddings, targets, labels)
    similarities = embeddings @ index.prototypes.T
    predictions = similarities.argmax(axis=1)
    accuracy = float((predictions == targets).mean())

    pairwise = 1.0 - embeddings @ embeddings.T
    identity = np.eye(len(targets), dtype=bool)
    same = (targets[:, None] == targets[None, :]) & ~identity
    different = targets[:, None] != targets[None, :]
    metrics = {
        "prototype_accuracy": accuracy,
        "mean_same_style_distance": float(pairwise[same].mean()),
        "mean_different_style_distance": float(pairwise[different].mean()),
    }
    return metrics, index


def verification_metrics(embeddings: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Evaluate pair verification on font identities never used for optimization."""

    distances = np.clip(1.0 - embeddings @ embeddings.T, 0.0, 2.0)
    rows, columns = np.triu_indices(len(targets), k=1)
    values = distances[rows, columns]
    different = targets[rows] != targets[columns]
    return {
        "heldout_pair_auroc": roc_auc(values, different),
        "heldout_same_style_distance": float(values[~different].mean()),
        "heldout_different_style_distance": float(values[different].mean()),
    }


def document_calibration_scores(
    encoder: StyleEncoder,
    fonts: Sequence[FontRecord],
    *,
    image_size: tuple[int, int],
    device: torch.device,
    documents: int = 24,
    seed: int = 17,
) -> np.ndarray:
    """Collect region-to-medoid distances from style-consistent synthetic pages.

    Conformal validity needs the calibration scores and the test scores to come from the
    same distribution. Word-crop groups do not: inference scores regions cut out of a
    rendered page by the OCR-free proposer, at page sizes and casings the crop sampler
    never sees. Calibrating through the deployed path instead keeps alpha meaningful.
    """

    probe = FontprintAnalyzer(
        encoder,
        DistanceCalibrator(np.array([0.0, 1.0], dtype=np.float32)),
        image_size=image_size,
        device=device,
    )
    scores: list[float] = []
    for step in range(documents):
        font = fonts[step % len(fonts)]
        document = render_document(font.path, None, tampered=False, seed=seed + step * 13)
        regions = propose_regions(document.image)
        if len(regions) < 3:
            continue
        embeddings = probe.embed_regions(document.image.convert("RGB"), regions)
        pairwise = np.clip(1.0 - embeddings @ embeddings.T, 0.0, 2.0)
        medoid = int(np.argmin(np.median(pairwise, axis=1)))
        distances = np.delete(pairwise[:, medoid], medoid)
        scores.extend(float(value) for value in distances)
    return np.asarray(scores, dtype=np.float32)


def _resolve_fonts(config: TrainConfig) -> tuple[list[FontRecord], list[FontRecord]]:
    fonts = discover_fonts(config.font_roots or None, limit=config.max_fonts + config.holdout_fonts)
    minimum = config.fonts_per_batch + config.holdout_fonts
    if len(fonts) < minimum:
        raise RuntimeError(
            f"found {len(fonts)} usable fonts, but training needs at least "
            f"{minimum}; pass additional --font-root directories"
        )
    holdout = fonts[-config.holdout_fonts :]
    train_fonts = fonts[: -config.holdout_fonts]
    return train_fonts, holdout


def train_model(
    config: TrainConfig,
    *,
    device: torch.device | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, float]]:
    """Train, calibrate, evaluate, and persist a complete inference bundle."""

    seed_everything(config.seed)
    target_device = device or preferred_device()
    fonts, holdout_fonts = _resolve_fonts(config)
    font_labels = [record.label for record in fonts]

    train_dataset = SyntheticFontDataset(
        fonts,
        config.samples_per_font,
        config.image_size,
        seed=config.seed,
        augment=True,
    )
    sampler = PKBatchSampler(
        len(fonts),
        config.samples_per_font,
        config.fonts_per_batch,
        config.samples_per_class,
        config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=target_device.type == "cuda",
    )
    validation_dataset = SyntheticFontDataset(
        fonts,
        config.validation_samples_per_font,
        config.image_size,
        seed=config.seed + 900_000,
        augment=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = TrainingModel(len(fonts), config.embedding_dim).to(target_device)
    contrastive_loss = SupervisedContrastiveLoss(config.temperature)
    classification_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets in train_loader:
            images, targets = images.to(target_device), targets.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(images)
            loss = contrastive_loss(embeddings, targets)
            loss = loss + config.classification_weight * classification_loss(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_result)
        if progress is not None:
            progress(epoch_result)

    embeddings, targets = collect_embeddings(model, validation_loader, target_device)
    metrics, index = retrieval_metrics(embeddings, targets, font_labels)
    holdout_dataset = SyntheticFontDataset(
        holdout_fonts,
        config.validation_samples_per_font,
        config.image_size,
        seed=config.seed + 1_800_000,
        augment=True,
    )
    holdout_loader = DataLoader(holdout_dataset, batch_size=config.batch_size, shuffle=False)
    holdout_embeddings, holdout_targets = collect_embeddings(model, holdout_loader, target_device)
    metrics.update(verification_metrics(holdout_embeddings, holdout_targets))

    calibration_dataset = SyntheticFontDataset(
        holdout_fonts,
        config.validation_samples_per_font,
        config.image_size,
        seed=config.seed + 2_700_000,
        augment=True,
    )
    calibration_loader = DataLoader(
        calibration_dataset, batch_size=config.batch_size, shuffle=False
    )
    calibration_embeddings, calibration_targets = collect_embeddings(
        model, calibration_loader, target_device
    )
    word_calibrator = DistanceCalibrator.fit(
        calibration_embeddings,
        calibration_targets,
        alpha=config.calibration_alpha,
        seed=config.seed,
    )
    page_scores = document_calibration_scores(
        model.encoder,
        holdout_fonts,
        image_size=config.image_size,
        device=target_device,
        documents=config.calibration_documents,
        seed=config.seed + 3_600_000,
    )
    # Prefer the page-matched null distribution; fall back only if it is unusably small.
    calibrator = (
        DistanceCalibrator(page_scores, config.calibration_alpha)
        if page_scores.size >= 32
        else word_calibrator
    )
    metrics["calibration_threshold"] = calibrator.threshold
    metrics["calibration_source"] = 1.0 if calibrator is not word_calibrator else 0.0
    metrics["calibration_samples"] = float(len(calibrator.scores))
    metrics["word_group_threshold"] = word_calibrator.threshold
    metrics["num_fonts"] = float(len(fonts))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "fontprint.pt"
    save_checkpoint(
        checkpoint_path,
        model.encoder,
        calibrator,
        index,
        image_size=config.image_size,
        metrics=metrics,
    )
    run_report = {
        "metrics": metrics,
        "history": history,
        "fonts": [{"label": font.label, "source_path": str(font.path)} for font in fonts],
        "holdout_fonts": [
            {"label": font.label, "source_path": str(font.path)} for font in holdout_fonts
        ],
        "config": config.model_dump(mode="json"),
        "device": str(target_device),
    }
    (config.output_dir / "run.json").write_text(
        json.dumps(run_report, indent=2) + "\n", encoding="utf-8"
    )
    return checkpoint_path, metrics
