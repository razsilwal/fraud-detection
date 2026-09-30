"""Reusable Streamlit UI components for FraudGuard."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

RISK_CLASS = {
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
    "CRITICAL": "critical",
}


def inject_css(css: str) -> None:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, pill: str | None = None) -> None:
    pill_html = f'<span class="fg-pill">{html.escape(pill)}</span>' if pill else ""
    st.markdown(
        f"""
        <div class="fg-hero">
            <h1>{html.escape(title)}{pill_html}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str = "", variant: str = "") -> None:
    """variant: '', 'danger', or 'success'."""
    cls = f"fg-card {variant}".strip()
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="label">{html.escape(label)}</div>
            <div class="value">{html.escape(value)}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> str:
    cat = (category or "LOW").upper()
    cls = RISK_CLASS.get(cat, "low")
    return f'<span class="fg-badge {cls}">{html.escape(cat)}</span>'


def risk_gauge(score: float, category: str) -> None:
    """Animated 0–100 risk meter."""
    score = max(0.0, min(100.0, float(score)))
    cat = (category or "LOW").upper()
    color = {
        "LOW": "#2ea043",
        "MEDIUM": "#d29922",
        "HIGH": "#c45c26",
        "CRITICAL": "#dc2626",
    }.get(cat, "#4F8BF9")

    st.markdown(
        f"""
        <div style="margin:6px 0 14px 0;">
          <div style="display:flex; justify-content:space-between; font-size:0.75rem;
                      color:#8b98a9; text-transform:uppercase; letter-spacing:0.08em;">
            <span>Risk score</span><span>{score:.0f}/100</span>
          </div>
          <div style="height:14px; width:100%; background:#161B22; border-radius:999px;
                      border:1px solid #24344d; overflow:hidden; margin-top:4px;">
            <div style="height:100%; width:{score}%; border-radius:999px;
                        background:linear-gradient(90deg, {color}88, {color});
                        transition: width 0.8s ease;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.7rem;
                      color:#6e7b8c; margin-top:4px;">
            <span>Safe</span><span>Watch</span><span>Block</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def transaction_form(sample: dict[str, Any] | None = None) -> dict[str, Any]:
    """Renders the transaction input form and returns a payload dict."""
    sample = sample or {}
    payload: dict[str, Any] = {}

    st.markdown("##### Transaction details")
    st.caption("PCA features (V1–V28) are anonymized by the dataset publisher. "
               "Use the sample buttons above or edit manually.")

    c1, c2, c3 = st.columns(3)
    with c1:
        payload["Time"] = st.number_input(
            "Time (seconds)", value=float(sample.get("Time", 0.0)), step=1.0
        )
    with c2:
        payload["Amount"] = st.number_input(
            "Amount", value=float(sample.get("Amount", 0.0)), min_value=0.0, step=1.0
        )
    with c3:
        payload["hour_of_day"] = int(sample.get("hour_of_day", 0))

    st.markdown("###### PCA components")
    with st.expander("V1 – V14", expanded=False):
        cols = st.columns(4)
        for i in range(1, 15):
            key = f"V{i}"
            with cols[(i - 1) % 4]:
                payload[key] = st.number_input(
                    key, value=float(sample.get(key, 0.0)), format="%.6f", key=f"in_{key}"
                )
    with st.expander("V15 – V28", expanded=False):
        cols = st.columns(4)
        for i in range(15, 29):
            key = f"V{i}"
            with cols[(i - 15) % 4]:
                payload[key] = st.number_input(
                    key, value=float(sample.get(key, 0.0)), format="%.6f", key=f"in_{key}"
                )
    return payload