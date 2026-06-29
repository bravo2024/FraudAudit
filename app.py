from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.core import compute_metrics
from src.visualizations import *
st.set_page_config(page_title="FraudAudit | KPMG AML/BSA", layout="wide", page_icon="\U0001f3e6")
with st.sidebar:
    st.header("\u2699 Config"); n = st.slider("Transactions", 5000, 50000, 20000, 1000)
    tau = st.slider("SAR Threshold", 0.05, 0.95, 0.50, 0.05)
    st.caption("KPMG | AML/BSA | Transaction Monitoring & SAR Advisory")
data = make_synthetic(n=n); b = train_all_models(data)
y_test = b["y_test"]; y_probas = {n: b["results"][n]["y_proba"] for n in b["results"]}
best = max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4 = st.columns(4)
c1.metric("Transactions",f"{n:,}"); c2.metric("SAR Rate",f"{data['positive_rate']:.2%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4 = st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f50d Transaction Monitoring","\U0001f4b0 SAR Filing"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax = plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Legitimate","Suspicious"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.4%}",ha="center",color="white")
    ax.set_title("SAR Filing Rate (highly imbalanced)",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.markdown("**AML indicators:** transaction_amount, transaction_velocity_24h, counterparty_risk_score, jurisdiction_risk_score, PEP flag, sanctions_match_score, KYC completeness, CDD tier, SAR filing history, unusual_pattern_flag")
with t2:
    rows = [{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b = st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test, y_probas))
    with col_b:
        st.markdown("#### Precision-Recall (PR-AUC preferred for imbalanced)")
        from sklearn.metrics import precision_recall_curve
        fig,ax = plt.subplots(figsize=(7,5)); _style()
        colors = ["#22d3ee","#a78bfa","#f97316","#f43f5e"]
        base = y_test.mean()
        for (name,y_p),c in zip(y_probas.items(),colors):
            pr, re, _ = precision_recall_curve(y_test, y_p)
            ax.plot(re, pr, color=c, lw=2, label=name)
        ax.axhline(base,color="#555",ls=":",lw=1.5)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curves",color="white")
        ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    st.pyplot(plot_confusion_matrix(y_test, b["results"]["XGBoost"]["y_pred"], "XGBoost"))
with t3:
    st.subheader("Transaction Monitoring — AML Alert Triage")
    st.latex(r"P(\text{SAR} \mid \text{Alert}) = \frac{P(\text{Alert} \mid \text{SAR})\,P(\text{SAR})}{P(\text{Alert})}")
    st.caption("Bayesian posterior that a flagged transaction warrants a Suspicious Activity Report. Prior P(SAR) is typically <0.01 in retail banking, higher in private banking.")
    st.latex(r"\text{CDD Tier} = \begin{cases} \text{Simplified} & \text{low-risk products} \\ \text{Standard} & \text{most customers} \\ \text{Enhanced} & \text{PEPs, high-risk jurisdictions} \end{cases}")
    st.caption("Customer Due Diligence tiering per FATF recommendations. EDD requires beneficial ownership identification, source-of-funds verification, and ongoing transaction monitoring.")
    st.latex(r"\text{Velocity Alert} = \mathbb{1}\!\left[\frac{N_{\text{txn}}(t-24h, t)}{\sigma_{N}} > 3\right]")
    st.caption("Transaction velocity threshold: alarms when 24-hour tx count exceeds 3 standard deviations from the customer's historical baseline. Red flags structuring/smurfing patterns.")
    st.markdown("**Rule-based + ML hybrid alerting:** Isolation Forest flags outliers for unstructured anomaly detection alongside supervised XGBoost. Alerts above threshold routed to AML analyst queue for SAR determination.")
    from sklearn.ensemble import IsolationForest
    X_num = data["X"][data["numerical_features"]].fillna(0)
    iso = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    iso_scores = iso.fit_transform(X_num)
    anomaly_flag = iso_scores < 0
    fig,ax = plt.subplots(figsize=(8,4)); _style()
    ax.hist(iso_scores, bins=50, color="#22d3ee", alpha=0.6)
    ax.axvline(0, color="#f43f5e", ls="--", lw=2, label="Anomaly threshold")
    ax.set_xlabel("Isolation Forest Score"); ax.set_ylabel("Count")
    ax.set_title("Unsupervised Anomaly Distribution — AML Alert Queue",color="white")
    ax.legend(); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.metric("Anomalies Detected (Isolation Forest)", anomaly_flag.sum())
    xgb_y = b["results"]["XGBoost"]["y_proba"]
    overlap = (anomaly_flag & (xgb_y > 0.5)).sum()
    st.metric("ML + IF Overlap (high-confidence SAR candidates)", overlap)
    st.metric("Estimated Analyst Review Hours", f"{anomaly_flag.sum() * 0.5:.0f}h")
with t4:
    st.subheader("SAR Filing Optimization")
    st.latex(r"\text{SAR Filing Threshold}(\tau) = \underset{\tau}{\arg\max} \; \text{TP}_{\text{SAR}} \times C_{\text{SAR}} - \text{FP} \times C_{\text{review}}")
    st.caption("Optimal SAR threshold balances filing all true suspicious activities (avoiding regulatory penalty) against overwhelming analysts with false-positive reviews.")
    st.latex(r"\text{Regulatory Penalty Risk} = \sum_{\text{missed SARs}} \text{Fine}_{\text{avg}} \approx \$2M\text{-}\$50M \text{ per violation (BSA/AML)}")
    st.caption("Failure to file SARs on reportable transactions can result in BSA/AML civil money penalties, consent orders, and in severe cases, charter revocation (e.g., Danske Bank \$2B, TD Bank \$3B).")
    avg_sar_value = st.number_input("Avg suspicious tx value ($)", 1000, 100000, 25000, 1000)
    review_cost = st.number_input("Cost per SAR review ($)", 50, 500, 150, 25)
    xgb_y = b["results"]["XGBoost"]["y_proba"]
    thresholds = np.linspace(0.05, 0.95, 91)
    results = []
    for tau in thresholds:
        yp = (xgb_y >= tau).astype(int)
        fn = ((y_test == 1) & (yp == 0)).sum()
        fp = ((y_test == 0) & (yp == 1)).sum()
        tp = ((y_test == 1) & (yp == 1)).sum()
        savings = tp * avg_sar_value - fp * review_cost
        results.append({"threshold": tau, "sar_filed": tp, "false_positives": fp, "net_value": savings, "false_positive_rate": fp / max(len(y_test),1)})
    res_df = pd.DataFrame(results)
    fig,ax = plt.subplots(figsize=(10,4)); _style()
    ax.plot(res_df["threshold"], res_df["net_value"], color="#22d3ee", lw=2)
    opt = res_df.loc[res_df["net_value"].idxmax()]
    ax.axvline(opt["threshold"], color="#f97316", ls="--", label=f"Optimal $\\tau$={opt['threshold']:.2f}")
    ax.set_xlabel("SAR Threshold"); ax.set_ylabel("Net Value Recovered ($)")
    ax.set_title("SAR Filing Optimization — Value vs Analyst Workload",color="white"); ax.legend(); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.metric("Max Net Value Recovered", f"${opt['net_value']:,.0f}")
    st.metric("SARs Filed", int(opt["sar_filed"]))
    st.metric("False Positives (analyst review)", int(opt["false_positives"]))
    st.metric("FP Rate", f"{opt['false_positive_rate']:.2%}")
