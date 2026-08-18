from __future__ import annotations

import numpy as np
import pytest

from fontprint.metrics import holm_adjusted, precision_recall_f1, roc_auc


def test_roc_auc_matches_hand_computed_values() -> None:
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert roc_auc(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 0, 1, 1])) == 0.5


def test_roc_auc_handles_ties_and_degenerate_labels() -> None:
    # Fully tied scores carry no ranking information, so the statistic must be 0.5.
    assert roc_auc(np.ones(6), np.array([0, 0, 0, 1, 1, 1])) == 0.5
    assert roc_auc(np.array([0.2, 0.4]), np.array([0, 0])) == 0.5
    with pytest.raises(ValueError):
        roc_auc(np.array([0.1, 0.2]), np.array([1]))


def test_precision_recall_f1_edges() -> None:
    precision, recall, f1 = precision_recall_f1(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (precision, recall) == (0.5, 0.5)
    assert f1 == pytest.approx(0.5)
    assert precision_recall_f1(np.zeros(4), np.ones(4)) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        precision_recall_f1(np.array([1, 0]), np.array([1]))


def test_holm_adjustment_matches_the_step_down_definition() -> None:
    adjusted = holm_adjusted(np.array([0.003, 0.01, 0.04, 0.5]))
    assert adjusted == pytest.approx([0.012, 0.03, 0.08, 0.5])


def test_holm_adjustment_is_monotone_and_order_preserving() -> None:
    raw = np.array([0.4, 0.02, 0.9])
    adjusted = holm_adjusted(raw)
    # The smallest raw p-value must stay the smallest adjusted one.
    assert int(np.argmin(adjusted)) == int(np.argmin(raw))
    assert np.all(adjusted >= raw)
    assert np.all(adjusted <= 1.0)
    # A single hypothesis needs no correction; an empty family is a no-op.
    assert holm_adjusted(np.array([0.02])) == pytest.approx([0.02])
    assert holm_adjusted(np.array([])).size == 0
