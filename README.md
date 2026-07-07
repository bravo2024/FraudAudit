# FraudAudit

Transaction monitoring that pairs a supervised classifier with unsupervised anomaly detection.

Trains four classifiers on synthetic transaction data to predict suspicious activity warranting SAR filing. Augments supervised learning with Isolation Forest for unsupervised anomaly detection in a hybrid alerting pipeline.

## Setup

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Detection tabs

| Tab | What it does |
|---|---|
| **Explorer** | Transaction records, SAR filing distribution (highly imbalanced), AML indicator descriptions |
| **Model Lab** | Multi-model comparison, ROC/PR curves (PR-AUC preferred for imbalanced data), confusion matrix, CV results |
| **Transaction Monitoring** | Bayesian SAR posterior, CDD tiering, velocity alerts, Isolation Forest anomaly distribution, ML + IF overlap analysis |
| **SAR Filing** | Optimal threshold optimisation, regulatory penalty risk calculator, cost-benefit analysis |

## Results

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.867 |
| Gini | 0.734 |
| KS Statistic | 0.598 |
| F1 Score | 0.357 |
| Accuracy | 0.800 |

5-fold CV AUC: 0.864 ± 0.005. Four models compared.

## Data

Synthetic transaction dataset: transaction amount, velocity (24h), counterparty risk score, jurisdiction risk score, PEP flag, sanctions match score, KYC completeness, CDD tier, and SAR filing history.

## Layout

```
FraudAudit/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## License

MIT
