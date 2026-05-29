from __future__ import annotations

import numpy as np

from evaluation.metrics import bookmaker_probabilities_without_margin, multiclass_brier


def test_bookmaker_probs_sum_to_one():
    odds = {"team": 2.20, "draw": 3.40, "opponent": 2.90}
    probs = bookmaker_probabilities_without_margin(odds)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["team"] > probs["opponent"]


def test_multiclass_brier_is_non_negative():
    y = np.array([0, 1, 2, 0])
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
        ]
    )
    score = multiclass_brier(y, probs, classes=[0, 1, 2])
    assert score >= 0.0
