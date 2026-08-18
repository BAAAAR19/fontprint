"""Compact convolutional style encoder used for metric learning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nnf


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nnf.gelu(self.body(inputs) + self.skip(inputs))


class StyleEncoder(nn.Module):
    """Map a normalized word crop to an L2-normalized style embedding."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            ResidualBlock(24, 48, stride=2),
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nnf.normalize(self.projection(self.features(inputs)), dim=1)


class TrainingModel(nn.Module):
    """Encoder plus a disposable classification head for hybrid supervision."""

    def __init__(self, num_classes: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = StyleEncoder(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(inputs)
        return embeddings, self.classifier(embeddings)
