from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def multiclass_brier(y_true, probabilities, classes) -> float:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_matrix = np.zeros_like(probabilities, dtype=float)
    for row_idx, label in enumerate(y_true):
        if label in class_to_idx:
            y_matrix[row_idx, class_to_idx[label]] = 1.0
    return float(np.mean(np.sum((probabilities - y_matrix) ** 2, axis=1)))


def evaluate_probabilistic_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    classes = model.classes_
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
        "brier_score": multiclass_brier(y.to_numpy(), probabilities, classes),
    }


def calibration_table(y_true, probabilities, classes, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(y_true)
    for class_idx, class_label in enumerate(classes):
        prob = probabilities[:, class_idx]
        bins = pd.cut(prob, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
        frame = pd.DataFrame({"bin": bins, "prob": prob, "actual": (y_true == class_label).astype(float)})
        grouped = frame.groupby("bin", observed=False).agg(
            mean_predicted=("prob", "mean"),
            observed_rate=("actual", "mean"),
            count=("actual", "size"),
        )
        grouped["class"] = class_label
        rows.append(grouped.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def bookmaker_probabilities_without_margin(odds: dict[str, float]) -> dict[str, float]:
    raw = {key: 1 / float(value) for key, value in odds.items()}
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}
