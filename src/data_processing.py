"""Load, inspect, split, and preprocess transaction data without leakage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import MODEL_FEATURE_COLUMNS, feature_frame, target_series
from src.utils import RAW_DATASET_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the credit-card CSV from disk."""
    csv_path = Path(path) if path is not None else Path(RAW_DATASET_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run `python scripts/download_data.py` or place creditcard.csv in data/raw/."
        )
    return pd.read_csv(csv_path)


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Compute inspection stats used by notebooks and the dashboard."""
    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    overview: dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "n_missing_cells": n_missing,
        "n_duplicate_rows": n_duplicates,
    }
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        n_fraud = int(counts.get(1, 0))
        n_legit = int(counts.get(0, 0))
        overview.update(
            {
                "n_fraud": n_fraud,
                "n_legitimate": n_legit,
                "fraud_rate": n_fraud / n_rows if n_rows else 0.0,
                "legitimate_rate": n_legit / n_rows if n_rows else 0.0,
            }
        )
    return overview


def drop_duplicates_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows.

    Duplicates can inflate apparent performance if the same transaction
    appears in both train and test. Removing them is justified here because
    they are identical copies, not independent observations.
    """
    return df.drop_duplicates().reset_index(drop=True)


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features/target with stratification so fraud rates match in both sets."""
    X = feature_frame(df)
    y = target_series(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """Scale all numeric model features.

    StandardScaler is fitted later on training data only (via Pipeline.fit).
    Fitting it on the full dataset would leak test-set means and variances.
    """
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), list(MODEL_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator: Any) -> Pipeline:
    """Wrap an estimator with the shared preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def positive_class_weight(y: pd.Series) -> float:
    """XGBoost scale_pos_weight ≈ count(negative) / count(positive)."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos
