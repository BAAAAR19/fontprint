"""Losses for font-style representation learning."""

from __future__ import annotations

import torch
from torch import nn


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss from Khosla et al., with stable log-sum-exp."""

    def __init__(self, temperature: float = 0.08) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [batch, embedding_dim]")
        labels = labels.reshape(-1)
        batch_size = embeddings.shape[0]
        identity = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        positives = labels[:, None].eq(labels[None, :]) & ~identity
        if not torch.all(positives.sum(dim=1) > 0):
            raise ValueError("every sample needs a positive pair; use PKBatchSampler")

        logits = embeddings @ embeddings.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits).masked_fill(identity, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_positive_log_prob = (log_prob * positives).sum(dim=1) / positives.sum(dim=1)
        return -mean_positive_log_prob.mean()
