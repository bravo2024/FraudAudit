"""Smoke tests for FraudAudit — unsupervised anomaly detection."""
from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import make_synthetic; from src.model import fit_and_evaluate
from src.core import precision_at_k, recall_at_k, mahalanobis_distance


def test_data():
    d = make_synthetic(200)
    assert d["n_samples"] == 200 and 0.01 < d["anomaly_rate"] < 0.15


def test_precision():
    assert precision_at_k([5, 4, 3, 2, 1], [1, 1, 0, 0, 0], 3) > 0.5


def test_recall():
    assert recall_at_k([5, 4, 3, 2, 1], [1, 1, 0, 0, 0], 3) > 0.3


def test_mahalanobis():
    import numpy as np
    X = np.array([[0, 0], [0, 0], [0, 0], [10, 10]])
    dists = mahalanobis_distance(X)
    assert dists[3] > dists[0]


def test_fit():
    d = make_synthetic(200)
    m, met = fit_and_evaluate(d)
    assert "iso_precision@50" in met


if __name__ == "__main__":
    test_data(); test_precision(); test_recall(); test_mahalanobis(); test_fit()
    print("All FraudAudit smoke tests passed!")
