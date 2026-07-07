"""core.py — Anomaly detection metrics for FraudAudit.

NOT generic classification. Anomaly detection for financial audit.
Metrics: Precision@k, Recall@k, Mahalanobis distance, audit coverage.
Reference: Liu et al. (2008) "Isolation Forest," ICDM.
"""
from __future__ import annotations
import numpy as np


def precision_at_k(anomaly_scores, true_labels, k):
    idx = np.argsort(-np.asarray(anomaly_scores, float))
    labels = np.asarray(true_labels, int)
    return float(labels[idx[:k]].sum() / k) if k > 0 else 0.0


def recall_at_k(anomaly_scores, true_labels, k):
    idx = np.argsort(-np.asarray(anomaly_scores, float))
    labels = np.asarray(true_labels, int)
    return float(labels[idx[:k]].sum() / max(labels.sum(), 1))


def audit_coverage(anomaly_scores, threshold=0.5):
    return float((np.asarray(anomaly_scores, float) >= threshold).mean())


def mahalanobis_distance(X):
    X = np.asarray(X, float)
    mu = X.mean(0)
    cov = np.cov(X.T, ddof=0)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    diffs = X - mu
    return np.array([float(np.sqrt(d @ inv @ d)) for d in diffs])
