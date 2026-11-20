"""Offline input-validation tests for cost-sensitive training."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.train import train


@pytest.mark.parametrize(
    "false_negative_cost,false_positive_cost",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (-float("inf"), 1.0),
        (5.0, float("nan")),
        (5.0, float("inf")),
        (5.0, -float("inf")),
        (0.0, 1.0),
        (5.0, 0.0),
        (-1.0, 1.0),
        (5.0, -1.0),
    ],
)
def test_invalid_costs_fail_before_dataset_fetch(false_negative_cost, false_positive_cost):
    with patch("src.train.load_data") as load_data:
        with pytest.raises(ValueError, match="finite positive"):
            train(Path("unused-output"), false_negative_cost, false_positive_cost, seed=42)
        load_data.assert_not_called()


def test_finite_positive_costs_reach_dataset_loader():
    with patch("src.train.load_data", side_effect=RuntimeError("loader reached")) as load_data:
        with pytest.raises(RuntimeError, match="loader reached"):
            train(Path("unused-output"), 5.0, 1.0, seed=42)
        load_data.assert_called_once_with()
