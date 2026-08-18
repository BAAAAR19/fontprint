"""Validated configuration shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class TrainConfig(BaseModel):
    seed: int = 17
    image_height: int = Field(64, ge=32)
    image_width: int = Field(160, ge=64)
    embedding_dim: int = Field(64, ge=16)
    font_roots: list[Path] = Field(default_factory=list)
    max_fonts: int = Field(16, ge=2)
    holdout_fonts: int = Field(4, ge=2)
    samples_per_font: int = Field(240, ge=4)
    validation_samples_per_font: int = Field(48, ge=2)
    fonts_per_batch: int = Field(8, ge=2)
    samples_per_class: int = Field(4, ge=2)
    epochs: int = Field(12, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    temperature: float = Field(0.08, gt=0)
    classification_weight: float = Field(0.35, ge=0)
    calibration_alpha: float = Field(0.05, gt=0, lt=1)
    calibration_documents: int = Field(24, ge=4)
    num_workers: int = Field(0, ge=0)
    output_dir: Path = Path("artifacts")

    @model_validator(mode="after")
    def validate_batch_shape(self) -> TrainConfig:
        if self.fonts_per_batch > self.max_fonts:
            raise ValueError("fonts_per_batch cannot exceed max_fonts")
        return self

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def batch_size(self) -> int:
        return self.fonts_per_batch * self.samples_per_class

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.model_validate(payload)
