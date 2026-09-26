"""Evaluation metrics, plots, and threshold analysis for imbalanced fraud detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import FIGURES_DIR


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the metrics that matter for rare-event detection.

    Accuracy is included only to demonstrate how misleading it can be.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Show how precision, recall, and error types change with the cutoff."""
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]
    rows = [classification_metrics(y_true, y_prob, t) for t in thresholds]
    return pd.DataFrame(rows)


def save_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    path: Path,
    title: str = "Confusion matrix",
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_prob, path: Path, title: str = "ROC curve") -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#1B3A4B", lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9aa3ab", ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_pr_curve(y_true, y_prob, path: Path, title: str = "Precision–Recall curve") -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#C45C26", lw=2, label=f"PR-AUC = {ap:.3f}")
    ax.axhline(baseline, color="#9aa3ab", ls="--", lw=1, label=f"Prevalence = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_threshold_plot(sweep: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["threshold"], sweep["precision"], marker="o", label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"], marker="o", label="Recall")
    ax.plot(sweep["threshold"], sweep["f1"], marker="o", label="F1")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold vs precision, recall, and F1")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a table without requiring the optional `tabulate` package."""

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(fmt(row[col]) for col in df.columns) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, divider, *body])


def metrics_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "Model": name,
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1": m.get("f1"),
                "ROC-AUC": m.get("roc_auc"),
                "PR-AUC": m.get("pr_auc"),
                "Accuracy": m.get("accuracy"),
            }
        )
    return pd.DataFrame(rows)
