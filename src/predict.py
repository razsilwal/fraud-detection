"""Inference: fraud probability, risk score, category, and local explanations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import MODEL_FEATURE_COLUMNS, feature_frame
from src.utils import (
    DEFAULT_THRESHOLD,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    MODELS_DIR,
)


def probability_to_risk_score(probability: float) -> int:
    """Map model probability in [0, 1] to an integer 0–100 score.

    This is a linear rescaling for communication. It is not a calibrated
    real-world probability unless calibration has been evaluated separately.
    """
    p = float(np.clip(probability, 0.0, 1.0))
    return int(round(p * 100))


def risk_category(score: int) -> str:
    if score <= LOW_RISK_MAX:
        return "LOW"
    if score <= MEDIUM_RISK_MAX:
        return "MEDIUM"
    return "HIGH"


def load_artifacts(models_dir: Path | None = None) -> dict[str, Any]:
    """Load the trained pipeline, optional explainer payload, and metadata."""
    root = models_dir or MODELS_DIR
    model_path = root / "model.pkl"
    meta_path = root / "model_metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Train first with `python -m src.train`."
        )
    pipeline = joblib.load(model_path)
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    preprocessor = None
    prep_path = root / "preprocessing.pkl"
    if prep_path.exists():
        preprocessor = joblib.load(prep_path)
    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "threshold": float(metadata.get("threshold", DEFAULT_THRESHOLD)),
        "feature_names": metadata.get("features", MODEL_FEATURE_COLUMNS),
    }


def _row_from_mapping(transaction: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(transaction, pd.DataFrame):
        df = transaction.copy()
    elif isinstance(transaction, pd.Series):
        df = transaction.to_frame().T
    else:
        df = pd.DataFrame([transaction])
    return feature_frame(df)


def predict_proba_row(pipeline: Any, transaction: dict[str, Any] | pd.Series | pd.DataFrame) -> float:
    X = _row_from_mapping(transaction)
    proba = pipeline.predict_proba(X)[0, 1]
    return float(np.clip(proba, 0.0, 1.0))


def local_feature_contributions(
    pipeline: Any,
    transaction: dict[str, Any] | pd.Series | pd.DataFrame,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Approximate local contributions using the model's feature importances

    scaled by standardized feature values. Tree SHAP is used in notebooks;
    this fallback keeps the app usable even if SHAP is slow or unavailable.
    """
    X = _row_from_mapping(transaction)
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    Xt = preprocess.transform(X)
    names = list(getattr(preprocess, "get_feature_names_out", lambda: MODEL_FEATURE_COLUMNS)())
    if hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        weights = np.abs(np.ravel(model.coef_))
    else:
        return []
    values = np.ravel(Xt)
    contrib = weights * np.abs(values)
    order = np.argsort(contrib)[::-1][:top_k]
    return [
        {
            "feature": str(names[i]) if i < len(names) else f"f{i}",
            "standardized_value": float(values[i]),
            "importance_weight": float(weights[i]),
            "contribution": float(contrib[i]),
        }
        for i in order
        if contrib[i] > 0
    ]


def predict_risk(
    transaction: dict[str, Any] | pd.Series | pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Return probability, 0–100 score, LOW/MEDIUM/HIGH category, and prediction."""
    arts = artifacts or load_artifacts()
    pipeline = arts["pipeline"]
    cutoff = arts["threshold"] if threshold is None else float(threshold)
    probability = predict_proba_row(pipeline, transaction)
    score = probability_to_risk_score(probability)
    category = risk_category(score)
    predicted_label = int(probability >= cutoff)
    return {
        "fraud_probability": probability,
        "risk_score": score,
        "risk_category": category,
        "model_prediction": predicted_label,
        "threshold": cutoff,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "contributing_features": local_feature_contributions(pipeline, transaction),
    }
