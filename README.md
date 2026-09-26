# FraudGuard

Fraud Detection & Risk Analytics Platform

A portfolio machine-learning system that scores credit-card transactions for fraud risk, explains the score, and documents the full imbalanced-classification lifecycle. It is built to be **technically honest**: strong numbers on a public PCA dataset are not the same thing as a production payment-risk engine.

## Overview

Payment fraud is a **rare-event classification** problem. Almost every transaction is legitimate, so a model that always predicts “not fraud” can report ~99.8% accuracy while catching nothing. FraudGuard treats **precision, recall, F1, ROC-AUC, and especially PR-AUC** as the evaluation language, converts model probability into a **0–100 risk score** and **LOW / MEDIUM / HIGH** bands, and serves the workflow in a Streamlit app.

## Motivation

Fraud detection is hard because:

- labels are scarce and delayed in real operations
- the two error types have asymmetric cost (**false negatives** often mean stolen funds)
- adversaries adapt (distribution shift)
- raw features are often confidential, so published research uses PCA components that are weakly interpretable

This repository shows how to handle those issues in a student-scale, reproducible project rather than a single `fit()` call.

## Features

- Structured EDA with charts that answer explicit questions
- Stratified splits and sklearn pipelines (no scaler fitted on test data)
- Imbalance handling compared experimentally (class weights, undersampling, SMOTE on **train folds only**)
- Baseline logistic regression plus tree ensembles (including XGBoost)
- Stratified cross-validation and randomized hyperparameter search scored with **average precision**
- Threshold analysis (0.1–0.9) instead of a frozen 0.5 cutoff
- Risk scoring distinct from raw probability
- Global/local explainability (SHAP in notebooks; contribution fallback in the app)
- Streamlit dashboard, analyzer, performance, analytics, and SQLite prediction history
- Automated tests for preprocessing and prediction contracts

## Dataset

**Credit Card Fraud Detection** — [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (also mirrored as the TensorFlow `creditcard.csv` tutorial file).

| Column | Meaning |
| --- | --- |
| `Time` | Seconds from the first transaction in the file |
| `V1`–`V28` | PCA components of confidential raw features |
| `Amount` | Transaction amount |
| `Class` | `1` fraud, `0` legitimate |

Place the CSV at `data/raw/creditcard.csv` (the download script does this for you). **Do not commit the CSV** — it is large. Stats in `reports/model_report.md` are computed from the file on disk, never invented.

### Dataset limitations

- Two days of traffic; not a longitudinal production log
- Anonymized PCA features block true domain explanations
- Class imbalance is extreme; small changes in threshold swing FP/FN counts
- Public labels are not the same as a bank’s chargeback process

## Machine learning pipeline

```
Data
 ↓
Cleaning (duplicate removal)
 ↓
EDA
 ↓
Deterministic feature engineering (hour-of-day proxy, log amount)
 ↓
Stratified train / test split
 ↓
Imbalance handling on training data only
 ↓
Model training (pipeline + scaler)
 ↓
Stratified cross-validation
 ↓
Hyperparameter tuning (average precision)
 ↓
Held-out test evaluation
 ↓
Threshold analysis + risk bands
 ↓
Explainability
 ↓
Serialized model → Streamlit
```

## Models

| Model | Role |
| --- | --- |
| Dummy (most frequent) | Accuracy trap: high accuracy, zero fraud recall |
| Logistic regression | Linear baseline; class weights vs resampling |
| Decision tree | Interpretable non-linear splits |
| Random forest | Bagged trees with class-balanced sampling |
| XGBoost | Boosted trees with `scale_pos_weight` |

The serialized app model is whichever candidate wins **PR-AUC** on the untouched test set after CV screening (see `models/model_metadata.json`).

## Evaluation

After you train, **actual metrics** are written to:

- `reports/model_report.md`
- `reports/test_metrics.csv`
- `models/model_metadata.json`

Those files are the only source of numbers suitable for a CV. Accuracy is reported as a cautionary metric, not a ranking metric.

## Screenshots

Run the app and capture:

1. Dashboard metric cards  
2. Risk analyzer with HIGH/MEDIUM/LOW badge  
3. Model performance (confusion matrix + PR curve)

Save images under `reports/figures/` if you publish the repo.

## Installation

Python **3.10–3.13** is recommended (3.13 used during development).

```bash
git clone <your-fork-url>
cd fraudguard

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python scripts/download_data.py
python -m src.train
```

Optional (Kaggle original instead of the public mirror): download `creditcard.csv` from [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`.

## Usage

```bash
streamlit run app/app.py
```

Jupyter:

```bash
python scripts/build_notebooks.py
jupyter notebook notebooks
```

Tests:

```bash
pytest -q
```

## Project structure

See the repository tree: `src/` training and inference, `app/` Streamlit UI, `notebooks/` learning walkthrough, `models/` joblib + metadata, `tests/` contracts.

## Limitations

- Not a real-time authorizer and not monitored for drift
- Probability is **not** claimed as a calibrated frequency unless you add calibration plots
- False positives still require an operations team this project does not simulate
- False negatives are cheap in a CSV and expensive in a bank
- SHAP on PCA features is “what the model used,” not a legal or causal explanation

## Future improvements

- Probability calibration (Platt / isotonic) with reliability diagrams
- Streaming inference and model monitoring / data drift
- Cost-sensitive thresholds from explicit FP vs FN dollar costs
- Autoencoders / isolation forest as complementary anomaly views
- Authenticated API (FastAPI) and cloud hosting
- Online learning if labels arrive with delay

## License

MIT — see `LICENSE`. Dataset terms remain those of the ULB / Kaggle publishers.
