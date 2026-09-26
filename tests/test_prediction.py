"""Prediction contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.predict import load_artifacts, predict_risk
from src.utils import MODELS_DIR


@pytest.mark.skipif(not (MODELS_DIR / "model.pkl").exists(), reason="Train the model before integration tests.")
def test_saved_model_probability_range() -> None:
    artifacts = load_artifacts()
    meta = artifacts["metadata"]
    features = meta.get("features") or []
    transaction = {name: 0.0 for name in ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]}
    transaction["Amount"] = 50.0
    transaction["Time"] = 1000.0
    result = predict_risk(transaction, artifacts=artifacts)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["risk_score"], int)


@pytest.mark.skipif(not (MODELS_DIR / "model_metadata.json").exists(), reason="No metadata yet.")
def test_metadata_has_required_keys() -> None:
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    for key in ("model_name", "features", "evaluation_metrics", "threshold", "dataset"):
        assert key in meta
    metrics = meta["evaluation_metrics"]
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert key in metrics
