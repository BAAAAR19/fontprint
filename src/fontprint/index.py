"""Small exact cosine index for interpretable nearest-style evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StyleMatch:
    label: str
    similarity: float


class PrototypeIndex:
    """Exact prototype search; intentionally dependency-free for small font catalogs."""

    def __init__(self, labels: list[str], prototypes: np.ndarray) -> None:
        values = np.asarray(prototypes, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(labels):
            raise ValueError("prototype shape must match labels")
        norms = np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
        self.labels = labels
        self.prototypes = values / norms

    @classmethod
    def fit(cls, embeddings: np.ndarray, targets: np.ndarray, labels: list[str]) -> PrototypeIndex:
        vectors = np.asarray(embeddings, dtype=np.float32)
        target_array = np.asarray(targets)
        prototypes = []
        for index in range(len(labels)):
            group = vectors[target_array == index]
            if not len(group):
                raise ValueError(f"no embeddings available for label {index}")
            prototypes.append(group.mean(axis=0))
        return cls(labels, np.stack(prototypes))

    def search(self, embedding: np.ndarray, k: int = 3) -> list[StyleMatch]:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-8)
        similarities = self.prototypes @ query
        indices = np.argsort(-similarities)[: min(k, len(self.labels))]
        return [StyleMatch(self.labels[index], float(similarities[index])) for index in indices]

    def to_dict(self) -> dict[str, object]:
        return {"labels": self.labels, "prototypes": self.prototypes.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PrototypeIndex:
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list):
            raise ValueError("index labels must be a list")
        labels = [str(value) for value in raw_labels]
        return cls(labels, np.asarray(payload["prototypes"], dtype=np.float32))
