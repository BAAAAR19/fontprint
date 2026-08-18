"""Small, dependency-light metric helpers shared by evaluation surfaces."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def roc_auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Rank-based AUROC that stays exact under tied scores.

    Equivalent to the Mann-Whitney U statistic, so it needs no threshold sweep and
    no scikit-learn dependency. Returns 0.5 when either class is empty, which keeps
    reports well-formed on degenerate splits instead of raising.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(positives).reshape(-1).astype(bool)
    if values.size != labels.size:
        raise ValueError("scores and positives must have the same length")
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return 0.5
    ranks = rankdata(values, method="average")
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def precision_recall_f1(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float, float]:
    """Binary precision, recall, and F1 with zero-division treated as zero."""

    guess = np.asarray(predicted).reshape(-1).astype(bool)
    truth = np.asarray(actual).reshape(-1).astype(bool)
    if guess.size != truth.size:
        raise ValueError("predicted and actual must have the same length")
    true_positive = float(np.count_nonzero(guess & truth))
    precision = true_positive / max(float(np.count_nonzero(guess)), 1.0)
    recall = true_positive / max(float(np.count_nonzero(truth)), 1.0)
    denominator = precision + recall
    f1 = 0.0 if denominator == 0 else 2 * precision * recall / denominator
    return precision, recall, f1
