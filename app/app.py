"""FraudGuard Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components import (
    hero,
    inject_css,
    metric_card,
    risk_badge,
    risk_gauge,
    transaction_form,
)
from app.styles import APP_CSS
from src.data_processing import dataset_overview, drop_duplicates_if_present, load_raw_dataset
from src.evaluate import classification_metrics, threshold_sweep
from src.feature_engineering import add_engineered_features
from src.history import load_history, save_prediction
from src.predict import load_artifacts, predict_risk
from src.utils import FIGURES_DIR, MODELS_DIR, RAW_DATASET_PATH

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    sample_path = ROOT / "data" / "processed" / "dashboard_sample.csv"
    if RAW_DATASET_PATH.exists():
        df = drop_duplicates_if_present(load_raw_dataset())
        return add_engineered_features(df)
    if sample_path.exists():
        return add_engineered_features(pd.read_csv(sample_path))
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    return load_artifacts()


def load_metadata() -> dict:
    path = MODELS_DIR / "model_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard(df: pd.DataFrame, meta: dict) -> None:
    hero(
        "FraudGuard",
        "Portfolio fraud-detection lab: imbalanced classification, thresholded risk scores, "
        "and explainable predictions on the public ULB credit-card dataset.",
    )
    if df.empty:
        st.warning(
            "No dataset found. Run `python scripts/download_data.py` then "
            "`python -m src.train` so the dashboard can load statistics."
        )
        return

    overview = dataset_overview(df)
    metrics = (meta.get("evaluation_metrics") or {}) if meta else {}
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Transactions", f"{overview['n_rows']:,}", hint="After dedup")
    with c2:
        metric_card(
            "Fraudulent",
            f"{overview.get('n_fraud', 0):,}",
            hint=f"{overview.get('fraud_rate', 0)*100:.3f}% of total",
            variant="danger",
        )
    with c3:
        metric_card(
            "Fraud rate",
            f"{overview.get('fraud_rate', 0)*100:.3f}%",
            hint="Rare-event problem",
        )
    with c4:
        avg_amt = float(df["Amount"].mean()) if "Amount" in df.columns else 0.0
        metric_card("Avg. amount", f"{avg_amt:,.2f}")
    with c5:
        pr = metrics.get("pr_auc")
        metric_card(
            "Model PR-AUC",
            f"{pr:.3f}" if pr is not None else "—",
            hint=meta.get("model_name", "not trained") if meta else "not trained",
            variant="success" if pr and pr > 0.5 else "",
        )

    st.markdown("##### What this page is for")
    st.write(
        "The cards above describe the **historical snapshot** used for research, "
        "not a live payment switch. Fraud is rare, so a naive accuracy number would "
        "look excellent while missing every fraudulent payment."
    )

    left, right = st.columns(2)
    with left:
        counts = (
            df["Class"]
            .map({0: "Legitimate", 1: "Fraud"})
            .value_counts()
            .rename_axis("status")
            .reset_index(name="count")
        )
        fig = px.bar(
            counts,
            x="status",
            y="count",
            color="status",
            color_discrete_map={"Legitimate": "#2c5364", "Fraud": "#c45c26"},
            title="Class counts (absolute)",
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        amount_sample = df.sample(n=min(len(df), 12000), random_state=42)
        fig = px.histogram(
            amount_sample,
            x="Amount",
            color=amount_sample["Class"].map({0: "Legitimate", 1: "Fraud"}),
            nbins=60,
            title="Amount distribution (sampled)",
            color_discrete_map={"Legitimate": "#2c5364", "Fraud": "#c45c26"},
        )
        fig.update_layout(
            legend_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    if "hour_of_day" in df.columns:
        hourly = (
            df.groupby(df["hour_of_day"].astype(int))["Class"]
            .mean()
            .reset_index()
            .rename(columns={"hour_of_day": "hour", "Class": "fraud_rate"})
        )
        fig = px.line(
            hourly,
            x="hour",
            y="fraud_rate",
            title="Fraud rate by hour-of-day proxy",
        )
        fig.update_traces(line_color="#c45c26")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    if metrics:
        st.markdown("##### Held-out test summary (from training run)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics.get('precision', float('nan')):.3f}")
        m2.metric("Recall", f"{metrics.get('recall', float('nan')):.3f}")
        m3.metric("F1", f"{metrics.get('f1', float('nan')):.3f}")
        m4.metric("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")
        st.caption(
            f"Selected model: {meta.get('model_name', 'n/a')} · "
            "Do not treat accuracy as the ranking metric on this dataset."
        )


# ---------------------------------------------------------------------------
# Page: Analyzer
# ---------------------------------------------------------------------------
def page_analyzer(df: pd.DataFrame, meta: dict) -> None:
    hero(
        "Transaction risk analyzer",
        "Score a single (demo) transaction. Probability comes from the saved model; "
        "the 0-100 score and LOW/MEDIUM/HIGH bands are communication layers, not calibrated odds.",
    )
    try:
        artifacts = load_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if "sample_row" not in st.session_state:
        st.session_state.sample_row = None

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Load legitimate sample", width="stretch") and not df.empty:
            legit = df[df["Class"] == 0]
            if not legit.empty:
                st.session_state.sample_row = (
                    legit.sample(1, random_state=None).iloc[0].to_dict()
                )
    with b2:
        if st.button("Load fraud sample", width="stretch") and not df.empty:
            fraud = df[df["Class"] == 1]
            if not fraud.empty:
                st.session_state.sample_row = (
                    fraud.sample(1, random_state=None).iloc[0].to_dict()
                )
    with b3:
        if st.button("Reset to zeros", width="stretch"):
            st.session_state.sample_row = {}

    threshold = st.slider(
        "Decision threshold (operational policy)",
        min_value=0.10,
        max_value=0.90,
        value=float(artifacts.get("threshold", 0.5)),
        step=0.05,
        help="Lower values catch more fraud (higher recall) and create more false alarms.",
    )
    payload = transaction_form(st.session_state.sample_row or {})

    if st.button("🔍 Predict risk", type="primary", width="stretch"):
        with st.spinner("Scoring transaction…"):
            result = predict_risk(payload, artifacts=artifacts, threshold=threshold)
        st.session_state.last_result = result
        st.session_state.last_payload = payload
        save_prediction(payload, result)

        cat = (result.get("risk_category") or "LOW").upper()
        if cat in ("HIGH", "CRITICAL"):
            st.toast(f"{cat} risk flagged", icon="🚨")
        elif cat == "MEDIUM":
            st.toast("Medium risk — review recommended", icon="⚠️")
        else:
            st.toast("Low risk — looks legitimate", icon="✅")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Submit a transaction to see probability, score, and contributing features.")
        return

    st.markdown('<div class="risk-panel">', unsafe_allow_html=True)
    a, b, c = st.columns([1.2, 1, 1])
    with a:
        metric_card(
            "Fraud probability",
            f"{result['fraud_probability']*100:.2f}%",
            hint=f"Threshold {result['threshold']:.2f}",
            variant="danger" if result["model_prediction"] == 1 else "",
        )
    with b:
        metric_card(
            "Risk score",
            f"{result['risk_score']}/100",
            hint="Communication layer",
            variant="danger" if result["risk_score"] >= 60 else "",
        )
    with c:
        st.markdown(
            '<div style="font-size:0.72rem;text-transform:uppercase;'
            'letter-spacing:0.08em;color:#8b98a9;margin-bottom:6px;">Risk level</div>',
            unsafe_allow_html=True,
        )
        st.markdown(risk_badge(result["risk_category"]), unsafe_allow_html=True)
        st.caption(
            f"Predicted class = **{result['model_prediction']}** "
            f"(prob ≥ {result['threshold']:.2f})"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    risk_gauge(result["risk_score"], result["risk_category"])

    st.markdown("##### Factors influencing this score")
    st.caption(
        "These are model attributions (importance × standardized value), not proof that a "
        "feature caused fraud. PCA components are not causal merchant attributes."
    )
    contrib = pd.DataFrame(result.get("contributing_features") or [])
    if contrib.empty:
        st.write("No feature contributions available for this estimator.")
    else:
        fig = px.bar(
            contrib.sort_values("contribution"),
            x="contribution",
            y="feature",
            orientation="h",
            title="Largest local contributions",
            color="contribution",
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(contrib, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page: Performance
# ---------------------------------------------------------------------------
def page_performance(meta: dict) -> None:
    hero(
        "Model performance",
        "Metrics on the untouched test split from the last training run. "
        "PR-AUC is emphasized because fraud is a rare event.",
    )
    payload_path = MODELS_DIR / "eval_payload.json"
    if not payload_path.exists() or not meta:
        st.warning(
            "Train the model first (`python -m src.train`) to populate evaluation artifacts."
        )
        return

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    y_true = np.asarray(payload["y_true"])
    y_prob = np.asarray(payload["y_prob"])
    threshold = st.slider(
        "Display threshold", 0.1, 0.9, float(payload.get("threshold", 0.5)), 0.05
    )
    metrics = classification_metrics(y_true, y_prob, threshold)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precision", f"{metrics['precision']:.3f}")
    k2.metric("Recall", f"{metrics['recall']:.3f}")
    k3.metric("F1", f"{metrics['f1']:.3f}")
    k4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    k5.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    st.markdown(
        f"**TP** {metrics['tp']} · **FP** {metrics['fp']} · "
        f"**FN** {metrics['fn']} · **TN** {metrics['tn']}  \n"
        f"Accuracy {metrics['accuracy']:.4f} is shown only as a cautionary number — "
        "a majority classifier would also look strong."
    )

    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual legit", "Actual fraud"],
        columns=["Pred legit", "Pred fraud"],
    )
    left, right = st.columns(2)
    with left:
        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion matrix",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        roc_png = FIGURES_DIR / "roc_curve.png"
        if roc_png.exists():
            st.image(str(roc_png), caption="ROC curve from training report")
    with right:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        rec, prec, _ = precision_recall_curve(y_true, y_prob)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        fig = px.line(roc_df, x="FPR", y="TPR", title="ROC curve")
        fig.update_traces(line_color="#1b3a4b")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        pr_df = pd.DataFrame({"Recall": rec, "Precision": prec})
        fig = px.line(pr_df, x="Recall", y="Precision", title="Precision-Recall curve")
        fig.update_traces(line_color="#c45c26")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    sweep = threshold_sweep(y_true, y_prob)
    fig = px.line(
        sweep,
        x="threshold",
        y=["precision", "recall", "f1"],
        title="Threshold analysis",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="score",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sweep, hide_index=True, width="stretch")

    table_path = ROOT / "reports" / "test_metrics.csv"
    if table_path.exists():
        st.markdown("##### All models on the same test split")
        st.dataframe(pd.read_csv(table_path), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------
def page_analytics(df: pd.DataFrame) -> None:
    hero(
        "Fraud analytics",
        "Exploratory views of amount, time-of-day proxy, and a few PCA components that "
        "often separate fraud from legitimate spend in this dataset.",
    )
    if df.empty:
        st.warning("Dataset not available.")
        return

    sample = df.sample(n=min(len(df), 15000), random_state=7)
    sample = sample.assign(status=sample["Class"].map({0: "Legitimate", 1: "Fraud"}))

    st.markdown("##### Are fraudulent amounts clustered?")
    fig = px.box(
        sample,
        x="status",
        y="Amount",
        color="status",
        points=False,
        color_discrete_map={"Legitimate": "#2c5364", "Fraud": "#c45c26"},
        title="Amount by class",
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Fraud is not confined to huge purchases; many fraudulent rows have modest amounts."
    )

    feat = st.selectbox(
        "PCA feature vs class (KDE-style histogram)",
        ["V4", "V10", "V12", "V14", "V17", "Amount"],
    )
    fig = px.histogram(
        sample,
        x=feat,
        color="status",
        nbins=70,
        histnorm="density",
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={"Legitimate": "#2c5364", "Fraud": "#c45c26"},
        title=f"Distribution of {feat}",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    corr_cols = [
        c
        for c in ["V4", "V10", "V12", "V14", "V17", "Amount", "hour_of_day", "Class"]
        if c in sample.columns
    ]
    corr = sample[corr_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation snapshot",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Correlation is association, not causation — especially for PCA components.")


# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------
def page_history() -> None:
    hero(
        "Prediction history",
        "Local SQLite log of demo scores from this machine. Do not store real cardholder data here.",
    )
    if st.button("🔄 Refresh"):
        st.rerun()
    hist = load_history()
    if hist.empty:
        st.info("No predictions stored yet. Score a transaction on the analyzer page.")
        return
    st.dataframe(
        hist.drop(columns=["payload_json"], errors="ignore"),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Raw payloads"):
        st.dataframe(hist, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css(APP_CSS)

    st.sidebar.markdown(
        """
        <div style="padding:6px 2px 12px 2px;">
          <div style="font-size:1.15rem; font-weight:700; color:#E6EDF3;">🛡️ FraudGuard</div>
          <div style="font-size:0.75rem; color:#8b98a9;">Fraud detection & risk analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "📊  Dashboard",
            "🔍  Transaction Risk Analyzer",
            "📈  Model Performance",
            "🧪  Fraud Analytics",
            "🗂️  Prediction History",
        ],
        label_visibility="collapsed",
    )
    page_key = page.split("  ", 1)[-1]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Educational portfolio system. High test-set scores on a public PCA dataset "
        "do not imply production readiness."
    )

    df = load_transactions()
    meta = load_metadata()

    if page_key == "Dashboard":
        page_dashboard(df, meta)
    elif page_key == "Transaction Risk Analyzer":
        page_analyzer(df, meta)
    elif page_key == "Model Performance":
        page_performance(meta)
    elif page_key == "Fraud Analytics":
        page_analytics(df)
    else:
        page_history()


if __name__ == "__main__":
    main()