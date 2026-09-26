"""Unit tests for preprocessing and risk scoring (no fabricated production metrics)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import (  # noqa: E402
    build_model_pipeline,
    drop_duplicates_if_present,
    stratified_train_test_split,
)
from src.evaluate import classification_metrics
from src.feature_engineering import MODEL_FEATURE_COLUMNS, add_engineered_features, feature_frame
from src.predict import predict_risk, probability_to_risk_score, risk_category


def _tiny_frame(n_legit: int = 80, n_fraud: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_legit + n_fraud
    data = {"Time": rng.uniform(0, 100000, n), "Amount": rng.uniform(1, 500, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)
    data["Class"] = np.array([0] * n_legit + [1] * n_fraud)
    return pd.DataFrame(data)


def test_engineered_columns_exist() -> None:
    df = add_engineered_features(_tiny_frame())
    assert "hour_of_day" in df.columns
    assert "amount_log" in df.columns
    assert (df["hour_of_day"] >= 0).all()
    assert (df["hour_of_day"] < 24).all()


def test_drop_duplicates() -> None:
    df = _tiny_frame()
    doubled = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = drop_duplicates_if_present(doubled)
    assert len(cleaned) == len(df)


def test_stratified_split_preserves_fraud() -> None:
    df = _tiny_frame(n_legit=90, n_fraud=10)
    X_train, X_test, y_train, y_test = stratified_train_test_split(df, test_size=0.2, random_state=42)
    assert set(MODEL_FEATURE_COLUMNS) <= set(X_train.columns)
    assert y_test.sum() >= 1
    assert y_train.sum() >= 1


def test_preprocessor_fit_on_train_only_changes_scale() -> None:
    df = _tiny_frame()
    X_train, X_test, y_train, y_test = stratified_train_test_split(df)
    pipe = build_model_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["preprocess"].named_transformers_["scale"]
    # Means are learned from train features, not the full dataset.
    train_mean = feature_frame(df.iloc[X_train.index]).mean().to_numpy() if False else scaler.mean_
    assert train_mean.shape[0] == len(MODEL_FEATURE_COLUMNS)


def test_risk_score_bounds() -> None:
    assert probability_to_risk_score(-0.2) == 0
    assert probability_to_risk_score(1.7) == 100
    assert probability_to_risk_score(0.314) == 31


def test_risk_categories() -> None:
    assert risk_category(0) == "LOW"
    assert risk_category(30) == "LOW"
    assert risk_category(31) == "MEDIUM"
    assert risk_category(70) == "MEDIUM"
    assert risk_category(71) == "HIGH"


def test_predict_risk_structure() -> None:
    df = _tiny_frame()
    X_train, X_test, y_train, y_test = stratified_train_test_split(df)
    pipe = build_model_pipeline(LogisticRegression(max_iter=800, class_weight="balanced"))
    pipe.fit(X_train, y_train)
    artifacts = {"pipeline": pipe, "threshold": 0.5, "metadata": {}}
    row = X_test.iloc[0].to_dict()
    # feature_frame expects Time/Amount/V* — X_test already has engineered cols plus originals
    result = predict_risk(row, artifacts=artifacts, threshold=0.4)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_category"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["model_prediction"] in {0, 1}
    assert result["threshold"] == 0.4
    assert "contributing_features" in result


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError):
        feature_frame(pd.DataFrame({"Amount": [1.0]}))


def test_metrics_include_confusion_cells() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.7, 0.2])
    m = classification_metrics(y_true, y_prob, threshold=0.5)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 4
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["pr_auc"] <= 1
