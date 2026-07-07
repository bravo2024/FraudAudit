"""model.py — Unsupervised anomaly detection for FraudAudit.

Three approaches: Isolation Forest, Mahalanobis distance, and Z-score ensemble.
This is UNSUPERVISED anomaly detection, NOT supervised classification.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from src.core import precision_at_k, recall_at_k, mahalanobis_distance, audit_coverage


def fit_and_evaluate(data, seed=42):
    X = data["X"]
    y = data["y"]

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=seed, n_jobs=-1)
    iso.fit(X)
    iso_scores = -iso.score_samples(X)

    # Mahalanobis distance
    maha = mahalanobis_distance(X)

    # Z-score ensemble (max absolute z-score across features)
    z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)).max(axis=1).ravel()

    metrics = {
        "n_samples": len(X),
        "anomaly_rate": float(y.mean()),
        "iso_precision@50": precision_at_k(iso_scores, y, 50),
        "iso_recall@50": recall_at_k(iso_scores, y, 50),
        "maha_precision@50": precision_at_k(maha, y, 50),
        "z_precision@50": precision_at_k(z_scores, y, 50),
        "audit_coverage": audit_coverage(iso_scores),
    }
    return {"models": {"iforest": iso}}, metrics