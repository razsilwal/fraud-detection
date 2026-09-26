"""SQLite history for Streamlit predictions (synthetic/demo inputs only)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import HISTORY_DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    risk_score INTEGER NOT NULL,
    risk_category TEXT NOT NULL,
    model_prediction INTEGER NOT NULL,
    threshold REAL NOT NULL
);
"""


def init_db(path: Path | None = None) -> Path:
    db_path = path or HISTORY_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()
    return db_path


def save_prediction(
    transaction: dict[str, Any],
    result: dict[str, Any],
    path: Path | None = None,
) -> None:
    db_path = init_db(path)
    # Store only numeric demo fields — never real cardholder data.
    safe_payload = {k: transaction.get(k) for k in sorted(transaction.keys())}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, payload_json, fraud_probability, risk_score,
                risk_category, model_prediction, threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("predicted_at") or datetime.now(timezone.utc).isoformat(),
                json.dumps(safe_payload, default=str),
                float(result["fraud_probability"]),
                int(result["risk_score"]),
                str(result["risk_category"]),
                int(result["model_prediction"]),
                float(result["threshold"]),
            ),
        )
        conn.commit()


def load_history(path: Path | None = None, limit: int = 200) -> pd.DataFrame:
    db_path = path or HISTORY_DB_PATH
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    return df
