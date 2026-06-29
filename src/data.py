from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = ["transaction_amount", "transaction_velocity_24h", "counterparty_risk_score", "jurisdiction_risk_score", "pep_flag", "sanctions_match_score", "kyc_complete_flag", "customer_tenure_days", "cdd_tier", "sar_filing_history", "unusual_pattern_flag", "transaction_type"]
CATEGORICAL_FEATURES = ["transaction_type", "cdd_tier"]
NUMERICAL_FEATURES = ["transaction_amount", "transaction_velocity_24h", "counterparty_risk_score", "jurisdiction_risk_score", "pep_flag", "sanctions_match_score", "kyc_complete_flag", "customer_tenure_days", "sar_filing_history", "unusual_pattern_flag"]

def make_synthetic(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "transaction_amount": rng.lognormal(mean=6.0, sigma=1.5, size=n).clip(100, 250000).round(2),
        "transaction_velocity_24h": rng.poisson(lam=5, size=n).clip(0, 50),
        "counterparty_risk_score": rng.uniform(1, 10, size=n).round(2),
        "jurisdiction_risk_score": rng.uniform(1, 10, size=n).round(2),
        "pep_flag": rng.choice([0, 1], size=n, p=[0.97, 0.03]),
        "sanctions_match_score": rng.beta(1, 20, size=n).round(4),
        "kyc_complete_flag": rng.choice([0, 1], size=n, p=[0.08, 0.92]),
        "customer_tenure_days": rng.exponential(scale=730, size=n).clip(1, 5000).astype(int),
        "cdd_tier": rng.choice(["simplified", "standard", "enhanced"], size=n, p=[0.20, 0.55, 0.25]),
        "sar_filing_history": rng.poisson(lam=0.3, size=n).clip(0, 5),
        "unusual_pattern_flag": rng.choice([0, 1], size=n, p=[0.90, 0.10]),
        "transaction_type": rng.choice(["wire", "ach", "check", "crypto", "cash_deposit"], size=n, p=[0.25, 0.30, 0.15, 0.10, 0.20]),
    })
    amount = np.log(df["transaction_amount"] + 1) / 12
    velocity = np.clip(df["transaction_velocity_24h"] / 50, 0, 1)
    cpty = df["counterparty_risk_score"] / 10
    juris = df["jurisdiction_risk_score"] / 10
    pep = df["pep_flag"]
    sanctions = 1 - np.exp(-10 * df["sanctions_match_score"])
    kyc = 1 - df["kyc_complete_flag"]
    tenure = np.clip(df["customer_tenure_days"] / 5000, 0, 1)
    cdd_map = {"simplified": 0.2, "standard": 0.5, "enhanced": 0.9}
    cdd = df["cdd_tier"].map(cdd_map).values
    sar = np.clip(df["sar_filing_history"] / 5, 0, 1)
    unusual = df["unusual_pattern_flag"]
    txn_map = {"wire": 0.3, "ach": 0.1, "check": 0.2, "crypto": 0.8, "cash_deposit": 0.5}
    txn = df["transaction_type"].map(txn_map).values
    log_odds = -4.5 + 0.3 * amount + 0.5 * velocity + 0.6 * cpty + 0.4 * juris + 0.8 * pep + 1.2 * sanctions + 0.5 * kyc - 0.3 * tenure + 0.4 * cdd + 0.6 * sar + 0.7 * unusual + 0.3 * txn + rng.normal(0, 0.6, size=n)
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > np.percentile(prob, 93)).astype(np.float64)
    return {"X": df, "y": y, "features": FEATURE_NAMES, "df": df.assign(suspicious_activity=y), "categorical_features": CATEGORICAL_FEATURES, "numerical_features": NUMERICAL_FEATURES, "n_samples": n, "n_features": len(FEATURE_NAMES), "positive_rate": y.mean()}
