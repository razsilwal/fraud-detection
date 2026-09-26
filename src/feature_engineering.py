"""
Feature construction for the credit-card fraud dataset.

The V1–V28 columns are already PCA components from the original (hidden)
transaction attributes. We add two interpretable transforms:

* hour_of_day — Time is seconds since the first transaction; wrapping by
  24 hours yields a cyclic daily clock that is easier to reason about.
* amount_log — log1p(Amount) compresses the long tail of large purchases.

These transforms are deterministic (no learned statistics), so they can be
applied before the train/test split without leaking labels or test-set
scale information. Scaling still happens later, fitted on train only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import PCA_FEATURE_COLUMNS, TARGET_COLUMN


ENGINEERED_COLUMNS = ["hour_of_day", "amount_log"]
MODEL_FEATURE_COLUMNS = [*PCA_FEATURE_COLUMNS, "Amount", "Time", *ENGINEERED_COLUMNS]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with hour_of_day and amount_log columns."""
    out = df.copy()
    if "Time" not in out.columns or "Amount" not in out.columns:
        missing = [c for c in ("Time", "Amount") if c not in out.columns]
        raise ValueError(f"Missing required columns for feature engineering: {missing}")
    out["hour_of_day"] = (out["Time"] % 86400) / 3600.0
    out["amount_log"] = np.log1p(out["Amount"].clip(lower=0))
    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select model input columns after engineering."""
    engineered = add_engineered_features(df)
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in engineered.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    return engineered[MODEL_FEATURE_COLUMNS]


def target_series(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")
    return df[TARGET_COLUMN].astype(int)
