"""Metrics for accuracy, calibration and selective prediction (abstention)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size.

    0 means the model's confidence can be taken at face value; a large value means
    '90% sure' does not mean 90% right.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def accuracy_at_coverage(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy when the model answers only its `coverage` most confident cases
    and abstains on (hands to a human) the rest."""
    k = max(1, int(round(coverage * len(confidence))))
    keep = np.argsort(-confidence)[:k]
    return float(correct[keep].mean())


def evaluate(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    pred = classes[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    correct = (pred == y_true).astype(float)
    acc = accuracy_score(y_true, pred)
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "mean_confidence": float(confidence.mean()),
        "overconfidence": float(confidence.mean() - acc),
        "ece": expected_calibration_error(confidence, correct),
        "acc_at_80pct_coverage": accuracy_at_coverage(confidence, correct, 0.8),
    }
