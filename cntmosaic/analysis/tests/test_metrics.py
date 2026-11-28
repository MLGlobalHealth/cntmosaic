"""Tests for the shared evaluator metrics (interval score and compute_metrics)."""

import numpy as np
import pytest

from cntmosaic.analysis.evaluator._base import compute_metrics, interval_score

# Interval [1, 3] with alpha = 0.05, so 2 / alpha = 40 and the width is 2.
LOW, HIGH, ALPHA = np.array([1.0]), np.array([3.0]), 0.05


@pytest.mark.parametrize(
    "y, expected",
    [
        (2.0, 2.0),  # inside: width only
        (1.0, 2.0),  # on the lower bound
        (3.0, 2.0),  # on the upper bound
        (0.0, 42.0),  # below: width + 40 * (1 - 0)
        (5.0, 82.0),  # above: width + 40 * (5 - 3)
    ],
)
def test_interval_score_cases(y, expected):
    assert interval_score(np.array([y]), LOW, HIGH, alpha=ALPHA) == pytest.approx(expected)


def test_interval_score_is_mean_over_elements():
    y = np.array([2.0, 0.0, 5.0])
    low, high = np.ones(3), np.full(3, 3.0)
    assert interval_score(y, low, high, alpha=ALPHA) == pytest.approx((2 + 42 + 82) / 3)


def test_interval_score_works_on_matrices():
    y = np.array([[2.0, 0.0], [5.0, 2.0]])
    low, high = np.ones((2, 2)), np.full((2, 2), 3.0)
    assert interval_score(y, low, high, alpha=ALPHA) == pytest.approx((2 + 42 + 82 + 2) / 4)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_interval_score_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        interval_score(np.ones(1), LOW, HIGH, alpha=alpha)


def test_compute_metrics_uses_alpha():
    y = np.array([0.0])
    # alpha = 0.1, so 2 / alpha = 20: width 2 + 20 * (1 - 0)
    assert compute_metrics(y, LOW, LOW, HIGH, alpha=0.1)[3] == pytest.approx(22.0)
    # default alpha = 0.05
    assert compute_metrics(y, LOW, LOW, HIGH)[3] == pytest.approx(42.0)


def test_aggregate_metrics_uses_alpha():
    from cntmosaic.analysis.evaluator._base import aggregate_metrics

    truth = {"A": np.array([[0.0]]), "B": np.array([[2.0]])}
    est = {k: np.stack([np.ones((1, 1)), np.full((1, 1), 2.0), np.full((1, 1), 3.0)]) for k in truth}
    df = aggregate_metrics(truth, est, alpha=0.1)
    scores = df.set_index("cat")["interval_score"]
    assert scores["A"] == pytest.approx(22.0)  # below: 2 + 20 * 1
    assert scores["B"] == pytest.approx(2.0)  # inside
