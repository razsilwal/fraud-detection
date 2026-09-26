"""Reusable Streamlit widgets for FraudGuard."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.utils import PCA_FEATURE_COLUMNS


def inject_css(css: str) -> None:
    st.markdown(css, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> str:
    key = category.lower()
    return f'<span class="badge badge-{key}">{category} RISK</span>'


def transaction_form(defaults: dict[str, Any] | None = None) -> dict[str, float]:
    """Collect Time, Amount, and PCA components for a single prediction."""
    defaults = defaults or {}
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=float(defaults.get("Amount", 88.0)),
            step=1.0,
            help="Original purchase amount in the dataset's currency units.",
        )
    with c2:
        time_value = st.number_input(
            "Time (seconds from first transaction)",
            min_value=0.0,
            value=float(defaults.get("Time", 40000.0)),
            step=1.0,
            help="Dataset clock — not a wall-clock timestamp from a live bank.",
        )

    payload: dict[str, float] = {"Amount": float(amount), "Time": float(time_value)}
    with st.expander("PCA components V1–V28 (anonymized features)", expanded=False):
        st.caption(
            "These columns are principal components of confidential raw fields. "
            "Zeros are a neutral demo default; use a sample row for realistic inputs."
        )
        cols = st.columns(4)
        for i, name in enumerate(PCA_FEATURE_COLUMNS):
            with cols[i % 4]:
                payload[name] = float(
                    st.number_input(
                        name,
                        value=float(defaults.get(name, 0.0)),
                        format="%.6f",
                        key=f"feat_{name}",
                    )
                )
    return payload
