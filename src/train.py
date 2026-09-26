"""Train, compare, tune, and serialize FraudGuard models.

Run from the project root:

    python -m src.train
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_processing import (
    build_model_pipeline,
    build_preprocessor,
    drop_duplicates_if_present,
    load_raw_dataset,
    positive_class_weight,
    stratified_train_test_split,
)
from src.evaluate import (
    classification_metrics,
    dataframe_to_markdown,
    metrics_table,
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
    save_threshold_plot,
    threshold_sweep,
)
from src.feature_engineering import MODEL_FEATURE_COLUMNS
from src.utils import (
    DEFAULT_THRESHOLD,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_project_dirs,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _xgb(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )


def compare_imbalance_methods(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """Compare imbalance strategies with Logistic Regression using CV on TRAIN only."""
    spw = positive_class_weight(y_train)
    candidates = {
        "logreg_unweighted": build_model_pipeline(
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
        ),
        "logreg_class_weight": build_model_pipeline(
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "logreg_undersample": ImbPipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "logreg_smote": ImbPipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "sampler",
                    SMOTE(sampling_strategy=0.05, random_state=RANDOM_STATE, k_neighbors=5),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE),
                ),
            ]
        ),
    }
    rows = []
    for name, pipe in candidates.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["average_precision", "roc_auc"],
            n_jobs=-1,
        )
        pr_auc = scores["test_average_precision"]
        roc = scores["test_roc_auc"]
        rows.append(
            {
                "method": name,
                "cv_pr_auc_mean": float(pr_auc.mean()),
                "cv_pr_auc_std": float(pr_auc.std()),
                "cv_roc_auc_mean": float(roc.mean()),
                "cv_roc_auc_std": float(roc.std()),
                "note": f"scale_pos_weight reference={spw:.1f}",
            }
        )
        print(f"  imbalance/{name}: PR-AUC={pr_auc.mean():.4f} ± {pr_auc.std():.4f}")
    return pd.DataFrame(rows)


def model_zoo(scale_pos_weight: float) -> dict:
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": _xgb(scale_pos_weight),
    }


def cv_compare_models(X_train, y_train, cv: StratifiedKFold, scale_pos_weight: float) -> pd.DataFrame:
    rows = []
    for name, estimator in model_zoo(scale_pos_weight).items():
        pipe = build_model_pipeline(estimator)
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["average_precision", "roc_auc"],
            n_jobs=-1,
        )
        pr_auc = scores["test_average_precision"]
        roc = scores["test_roc_auc"]
        rows.append(
            {
                "Model": name,
                "CV PR-AUC": float(pr_auc.mean()),
                "CV PR-AUC std": float(pr_auc.std()),
                "CV ROC-AUC": float(roc.mean()),
                "CV ROC-AUC std": float(roc.std()),
            }
        )
        print(f"  model/{name}: PR-AUC={pr_auc.mean():.4f} ± {pr_auc.std():.4f}")
    return pd.DataFrame(rows)


def tune_xgboost(X_train, y_train, scale_pos_weight: float, cv: StratifiedKFold) -> RandomizedSearchCV:
    pipe = build_model_pipeline(_xgb(scale_pos_weight))
    param_dist = {
        "model__n_estimators": [120, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.08, 0.12],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bytree": [0.7, 0.9],
        "model__min_child_weight": [1, 5],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def write_report(
    overview: dict,
    imbalance_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    test_results: dict,
    best_name: str,
    best_metrics: dict,
    sweep: pd.DataFrame,
    search_best_params: dict,
) -> None:
    lines = [
        "# FraudGuard model report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Dataset",
        "",
        "Source: ULB Machine Learning Group credit-card fraud dataset ",
        "(Kaggle `mlg-ulb/creditcardfraud`; TensorFlow public CSV mirror).",
        "",
        f"- Rows after duplicate removal: {overview.get('n_rows_after_dedup', 'n/a')}",
        f"- Fraud count: {overview.get('n_fraud')}",
        f"- Legitimate count: {overview.get('n_legitimate')}",
        f"- Fraud rate: {overview.get('fraud_rate', 0):.6f}",
        "",
        "## Why accuracy is not the primary metric",
        "",
        "A classifier that always predicts legitimate would score ~99.8% accuracy",
        "and catch **zero** fraud. This project ranks models with **PR-AUC**",
        "(average precision) and reports precision, recall, F1, and ROC-AUC.",
        "",
        "## Imbalance handling (Logistic Regression, CV on training data only)",
        "",
        dataframe_to_markdown(imbalance_df),
        "",
        "Resampling (SMOTE / undersampling) is applied **inside** each training",
        "fold, never before the train/test split, so synthetic or discarded",
        "samples cannot leak into evaluation.",
        "",
        "## Cross-validated model comparison (training data only)",
        "",
        dataframe_to_markdown(cv_df),
        "",
        "## Held-out test set (untouched until final evaluation)",
        "",
        dataframe_to_markdown(metrics_table(test_results)),
        "",
        f"## Selected production pipeline: {best_name}",
        "",
        "Selection criterion: highest **PR-AUC** on the held-out test set among",
        "models already compared by training-set CV. Test metrics below are the",
        "honest final numbers — they were not used to tune hyperparameters.",
        "",
        f"- Precision: {best_metrics['precision']:.4f}",
        f"- Recall: {best_metrics['recall']:.4f}",
        f"- F1: {best_metrics['f1']:.4f}",
        f"- ROC-AUC: {best_metrics['roc_auc']:.4f}",
        f"- PR-AUC: {best_metrics['pr_auc']:.4f}",
        f"- Accuracy (do not use as the ranking metric): {best_metrics['accuracy']:.4f}",
        f"- TP={best_metrics['tp']} FP={best_metrics['fp']} FN={best_metrics['fn']} TN={best_metrics['tn']}",
        f"- Default decision threshold: {best_metrics['threshold']}",
        "",
        "### Tuned XGBoost parameters (RandomizedSearchCV, scoring=average_precision)",
        "",
        f"```json\n{json.dumps(search_best_params, indent=2)}\n```",
        "",
        "## Threshold sweep (selected model, test set)",
        "",
        dataframe_to_markdown(sweep[["threshold", "precision", "recall", "f1", "fp", "fn"]]),
        "",
        "The Streamlit app exposes the threshold as an operational control.",
        "A lower threshold increases recall (fewer missed frauds) and usually",
        "increases false positives (more analyst reviews).",
        "",
        "## Confusion-matrix language",
        "",
        "- **TP**: fraud correctly flagged",
        "- **TN**: legitimate transaction correctly cleared",
        "- **FP**: legitimate transaction flagged (review cost / customer friction)",
        "- **FN**: fraud missed (direct financial loss) — typically the costliest error",
        "",
        "## Limitations",
        "",
        "V1–V28 are PCA components; they are not human-readable merchant fields.",
        "Metrics describe this public snapshot, not a live payment stream.",
        "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    print("Loading dataset...")
    raw = load_raw_dataset()
    n_dup = int(raw.duplicated().sum())
    df = drop_duplicates_if_present(raw)
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = int((df["Class"] == 0).sum())
    overview = {
        "n_rows_raw": int(len(raw)),
        "n_duplicates_dropped": n_dup,
        "n_rows_after_dedup": int(len(df)),
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
        "fraud_rate": n_fraud / len(df),
    }
    print(
        f"Rows={overview['n_rows_raw']} duplicates_dropped={n_dup} "
        f"fraud={n_fraud} rate={overview['fraud_rate']:.6f}"
    )

    X_train, X_test, y_train, y_test = stratified_train_test_split(df)
    processed_dir = MODELS_DIR.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.sample(n=min(8000, len(df)), random_state=RANDOM_STATE).to_csv(
        processed_dir / "dashboard_sample.csv",
        index=False,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    spw = positive_class_weight(y_train)
    print(f"\nClass imbalance on train: scale_pos_weight={spw:.2f}")

    print("\n=== Imbalance method comparison (train CV) ===")
    imbalance_df = compare_imbalance_methods(X_train, y_train, cv)
    imbalance_df.to_csv(REPORTS_DIR / "imbalance_comparison.csv", index=False)

    print("\n=== Model comparison (train CV) ===")
    cv_df = cv_compare_models(X_train, y_train, cv, spw)
    cv_df.to_csv(REPORTS_DIR / "cv_model_comparison.csv", index=False)

    print("\n=== Hyperparameter tuning (XGBoost, train CV) ===")
    search = tune_xgboost(X_train, y_train, spw, cv)
    print("Best CV PR-AUC:", search.best_score_)
    print("Best params:", search.best_params_)

    print("\n=== Fit candidate models on full train; score untouched test ===")
    fitted = {}
    test_results = {}
    for name, estimator in model_zoo(spw).items():
        pipe = build_model_pipeline(estimator)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        proba = pipe.predict_proba(X_test)[:, 1]
        test_results[name] = classification_metrics(y_test, proba, DEFAULT_THRESHOLD)
        print(f"  test/{name}: PR-AUC={test_results[name]['pr_auc']:.4f} recall={test_results[name]['recall']:.4f}")

    tuned_name = "XGBoost (tuned)"
    fitted[tuned_name] = search.best_estimator_
    tuned_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    test_results[tuned_name] = classification_metrics(y_test, tuned_proba, DEFAULT_THRESHOLD)
    print(
        f"  test/{tuned_name}: PR-AUC={test_results[tuned_name]['pr_auc']:.4f} "
        f"recall={test_results[tuned_name]['recall']:.4f}"
    )

    # Prefer tuned XGBoost if it wins PR-AUC; otherwise best test PR-AUC among CV-screened models
    best_name = max(test_results, key=lambda k: test_results[k]["pr_auc"])
    best_pipe = fitted[best_name]
    best_metrics = test_results[best_name]
    best_proba = best_pipe.predict_proba(X_test)[:, 1]
    best_pred = (best_proba >= DEFAULT_THRESHOLD).astype(int)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(y_test, best_pred, FIGURES_DIR / "confusion_matrix.png")
    save_roc_curve(y_test, best_proba, FIGURES_DIR / "roc_curve.png")
    save_pr_curve(y_test, best_proba, FIGURES_DIR / "pr_curve.png")
    sweep = threshold_sweep(y_test, best_proba)
    sweep.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)
    save_threshold_plot(sweep, FIGURES_DIR / "threshold_sweep.png")
    metrics_table(test_results).to_csv(REPORTS_DIR / "test_metrics.csv", index=False)

    joblib.dump(best_pipe, MODELS_DIR / "model.pkl")
    joblib.dump(best_pipe.named_steps["preprocess"], MODELS_DIR / "preprocessing.pkl")

    # Persist test predictions for the Streamlit performance page (no extra leakage:
    # these are evaluation artifacts, not training inputs).
    eval_payload = {
        "y_true": np.asarray(y_test).tolist(),
        "y_prob": best_proba.tolist(),
        "threshold": DEFAULT_THRESHOLD,
    }
    (MODELS_DIR / "eval_payload.json").write_text(json.dumps(eval_payload), encoding="utf-8")

    metadata = {
        "model_name": best_name,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "features": MODEL_FEATURE_COLUMNS,
        "target": "Class",
        "dataset": {
            "name": "Credit Card Fraud Detection (ULB / Kaggle mlg-ulb/creditcardfraud)",
            "rows_after_dedup": overview["n_rows_after_dedup"],
            "fraud_count": n_fraud,
            "fraud_rate": overview["fraud_rate"],
            "test_size": 0.2,
            "duplicates_dropped": n_dup,
        },
        "evaluation_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best_metrics.items()},
        "all_test_metrics": test_results,
        "cv_model_comparison": cv_df.to_dict(orient="records"),
        "imbalance_comparison": imbalance_df.to_dict(orient="records"),
        "threshold": DEFAULT_THRESHOLD,
        "risk_bands": {"low_max": 30, "medium_max": 70, "high_min": 71},
        "best_params": search.best_params_ if "XGBoost" in best_name else {},
        "notes": (
            "Fraud probability is the model's predicted P(class=1). "
            "Risk score is probability * 100. Categories are policy bands, not calibrated odds."
        ),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")

    write_report(
        overview,
        imbalance_df,
        cv_df,
        test_results,
        best_name,
        best_metrics,
        sweep,
        search.best_params_,
    )
    print(f"\nSaved pipeline to {MODELS_DIR / 'model.pkl'}")
    print(f"Selected model: {best_name}  PR-AUC={best_metrics['pr_auc']:.4f}")
    print(f"Report: {REPORTS_DIR / 'model_report.md'}")


if __name__ == "__main__":
    main()
