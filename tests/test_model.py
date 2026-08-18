from __future__ import annotations

import pytest
import torch

from fontprint.losses import SupervisedContrastiveLoss
from fontprint.model import StyleEncoder, TrainingModel


def test_encoder_returns_unit_embeddings() -> None:
    model = StyleEncoder(embedding_dim=16).eval()
    outputs = model(torch.rand(3, 1, 64, 160))
    assert outputs.shape == (3, 16)
    assert torch.allclose(outputs.norm(dim=1), torch.ones(3), atol=1e-5)


def test_hybrid_model_and_contrastive_loss_backpropagate() -> None:
    model = TrainingModel(num_classes=2, embedding_dim=16)
    embeddings, logits = model(torch.rand(4, 1, 64, 160))
    loss = SupervisedContrastiveLoss()(embeddings, torch.tensor([0, 0, 1, 1]))
    loss = loss + torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.projection[-1].weight.grad is not None


def test_contrastive_loss_rejects_batches_without_positives() -> None:
    with pytest.raises(ValueError, match="positive pair"):
        SupervisedContrastiveLoss()(torch.eye(3), torch.tensor([0, 1, 2]))
