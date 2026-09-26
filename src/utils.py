"""Shared paths, constants, and small helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = PROJECT_ROOT / "app"

RAW_DATASET_PATH = DATA_RAW / "creditcard.csv"
DATASET_FILENAME = "creditcard.csv"

# Public mirror of the ULB Machine Learning Group credit-card fraud dataset
# (same file used in TensorFlow tutorials). Original source: Kaggle / ULB.
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

TARGET_COLUMN = "Class"
PCA_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURE_COLUMNS = ["Time", "Amount", *PCA_FEATURE_COLUMNS]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Operational risk bands applied to a 0–100 score derived from model probability.
# These bands are policy choices, not calibrated real-world probabilities.
LOW_RISK_MAX = 30
MEDIUM_RISK_MAX = 70

DEFAULT_THRESHOLD = 0.50

HISTORY_DB_PATH = PROJECT_ROOT / "data" / "prediction_history.sqlite"


def ensure_project_dirs() -> None:
    """Create folders that are required at runtime."""
    for path in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
