"""Tests for exact validation-cost threshold selection."""

import numpy as np
import pandas as pd
import pytest

from src.train import choose_threshold


def test_exact_cutoff_between_grid_points_avoids_misclassification():
    labels = pd.Series([0, 1])
    threshold, cost = choose_threshold(labels, np.array([0.503, 0.504]), 5, 1)
    assert threshold == pytest.approx(0.504)
    assert cost == 0


def test_all_negative_predictions_can_be_optimal_even_with_score_one():
    labels = pd.Series([0, 1])
    threshold, cost = choose_threshold(labels, np.array([1.0, 0.9]), 1, 100)
    assert threshold > 1.0
    assert cost == 1


def test_tied_probabilities_choose_smallest_equal_cost_threshold():
    threshold, cost = choose_threshold(pd.Series([0, 1]), np.array([0.4, 0.4]), 1, 1)
    assert threshold == pytest.approx(0.4)
    assert cost == 1


@pytest.mark.parametrize("scores", [[0.2, np.nan], [-0.1, 0.5], [0.2, 1.1]])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError, match="probabilities must be finite"):
        choose_threshold(pd.Series([0, 1]), np.asarray(scores), 5, 1)


def test_mismatched_input_lengths_rejected():
    with pytest.raises(ValueError, match="aligned vectors"):
        choose_threshold(pd.Series([0, 1]), np.array([0.2]), 5, 1)
