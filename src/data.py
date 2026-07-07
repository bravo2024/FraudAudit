"""data.py — Synthetic audit data with injected anomalies.

Multivariate normal data with 5% anomalies (shifted means).
This is anomaly detection data, NOT binary classification.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


def make_synthetic(n: int = 2000, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nf = 10
    X = rng.normal(0, 1, (n, nf))
    anom = rng.random(n) < 0.05
    X[anom] += rng.normal(3, 1.5, (anom.sum(), nf))
    names = [f"txn_amount_{i}" for i in range(nf)]
    df = pd.DataFrame(X, columns=names)
    return {
        "X": X, "y": anom.astype(int), "df": df,
        "n_samples": n, "n_features": nf,
        "anomaly_rate": float(anom.mean()),
    }