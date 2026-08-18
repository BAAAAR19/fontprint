"""Finite-sample split-conformal calibration for open-set distance scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DistanceCalibrator:
    """Calibration distribution of query-to-medoid same-style distances."""

    scores: np.ndarray
    alpha: float = 0.05

    def __post_init__(self) -> None:
        clean = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if clean.size < 2:
            raise ValueError("calibration requires at least two scores")
        if not np.all(np.isfinite(clean)):
            raise ValueError("calibration scores must be finite")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")
        object.__setattr__(self, "scores", np.sort(clean))

    @property
    def threshold(self) -> float:
        rank = int(np.ceil((len(self.scores) + 1) * (1.0 - self.alpha))) - 1
        return float(self.scores[min(max(rank, 0), len(self.scores) - 1)])

    def p_value(self, score: float) -> float:
        """Probability of seeing a same-style distance at least this extreme."""

        extreme = int(np.count_nonzero(self.scores >= score))
        return (extreme + 1.0) / (len(self.scores) + 1.0)

    def is_anomaly(self, score: float) -> bool:
        return self.p_value(score) <= self.alpha

    def to_dict(self) -> dict[str, object]:
        return {"scores": self.scores.tolist(), "alpha": self.alpha}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DistanceCalibrator:
        alpha = payload.get("alpha")
        if not isinstance(alpha, (int, float)):
            raise ValueError("calibration alpha must be numeric")
        return cls(np.asarray(payload["scores"], dtype=np.float32), float(alpha))

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        alpha: float = 0.05,
        reference_size: int = 3,
        max_groups_per_class: int = 64,
        seed: int = 17,
    ) -> DistanceCalibrator:
        """Build disjoint query/reference groups that mirror document inference."""

        vectors = np.asarray(embeddings, dtype=np.float32)
        targets = np.asarray(labels)
        rng = np.random.default_rng(seed)
        distances: list[float] = []
        for label in np.unique(targets):
            class_vectors = vectors[targets == label].copy()
            rng.shuffle(class_vectors, axis=0)
            group_size = reference_size + 1
            group_count = min(len(class_vectors) // group_size, max_groups_per_class)
            if group_count == 0:
                continue
            for group_index in range(group_count):
                start = group_index * group_size
                group = class_vectors[start : start + group_size]
                references, query = group[:-1], group[-1]
                pairwise = np.clip(1.0 - references @ references.T, 0.0, 2.0)
                medoid = references[int(np.argmin(np.median(pairwise, axis=1)))]
                distances.append(float(np.clip(1.0 - query @ medoid, 0.0, 2.0)))
        return cls(np.asarray(distances, dtype=np.float32), alpha)
