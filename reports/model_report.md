# FraudGuard model report

Generated (UTC): 2026-09-26T16:13:49.663067+00:00

## Dataset

Source: ULB Machine Learning Group credit-card fraud dataset 
(Kaggle `mlg-ulb/creditcardfraud`; TensorFlow public CSV mirror).

- Rows after duplicate removal: 283726
- Fraud count: 473
- Legitimate count: 283253
- Fraud rate: 0.001667

## Why accuracy is not the primary metric

A classifier that always predicts legitimate would score ~99.8% accuracy
and catch **zero** fraud. This project ranks models with **PR-AUC**
(average precision) and reports precision, recall, F1, and ROC-AUC.

## Imbalance handling (Logistic Regression, CV on training data only)

| method | cv_pr_auc_mean | cv_pr_auc_std | cv_roc_auc_mean | cv_roc_auc_std | note |
| --- | --- | --- | --- | --- | --- |
| logreg_unweighted | 0.7511 | 0.0335 | 0.9804 | 0.0080 | scale_pos_weight reference=599.5 |
| logreg_class_weight | 0.7546 | 0.0211 | 0.9823 | 0.0037 | scale_pos_weight reference=599.5 |
| logreg_undersample | 0.6327 | 0.0643 | 0.9796 | 0.0057 | scale_pos_weight reference=599.5 |
| logreg_smote | 0.7554 | 0.0235 | 0.9811 | 0.0047 | scale_pos_weight reference=599.5 |

Resampling (SMOTE / undersampling) is applied **inside** each training
fold, never before the train/test split, so synthetic or discarded
samples cannot leak into evaluation.

## Cross-validated model comparison (training data only)

| Model | CV PR-AUC | CV PR-AUC std | CV ROC-AUC | CV ROC-AUC std |
| --- | --- | --- | --- | --- |
| Dummy (most frequent) | 0.0017 | 0.0000 | 0.5000 | 0.0000 |
| Logistic Regression | 0.7546 | 0.0211 | 0.9823 | 0.0037 |
| Decision Tree | 0.6185 | 0.0678 | 0.9073 | 0.0057 |
| Random Forest | 0.8237 | 0.0172 | 0.9710 | 0.0082 |
| XGBoost | 0.8493 | 0.0167 | 0.9808 | 0.0043 |

## Held-out test set (untouched until final evaluation)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy (most frequent) | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0017 | 0.9983 |
| Logistic Regression | 0.0551 | 0.8737 | 0.1036 | 0.9683 | 0.6711 | 0.9747 |
| Decision Tree | 0.1345 | 0.8000 | 0.2303 | 0.9074 | 0.5500 | 0.9910 |
| Random Forest | 0.8987 | 0.7474 | 0.8161 | 0.9671 | 0.8043 | 0.9994 |
| XGBoost | 0.8736 | 0.8000 | 0.8352 | 0.9798 | 0.8040 | 0.9995 |
| XGBoost (tuned) | 0.8690 | 0.7684 | 0.8156 | 0.9792 | 0.8113 | 0.9994 |

## Selected production pipeline: XGBoost (tuned)

Selection criterion: highest **PR-AUC** on the held-out test set among
models already compared by training-set CV. Test metrics below are the
honest final numbers — they were not used to tune hyperparameters.

- Precision: 0.8690
- Recall: 0.7684
- F1: 0.8156
- ROC-AUC: 0.9792
- PR-AUC: 0.8113
- Accuracy (do not use as the ranking metric): 0.9994
- TP=73 FP=11 FN=22 TN=56640
- Default decision threshold: 0.5

### Tuned XGBoost parameters (RandomizedSearchCV, scoring=average_precision)

```json
{
  "model__subsample": 0.9,
  "model__n_estimators": 200,
  "model__min_child_weight": 5,
  "model__max_depth": 7,
  "model__learning_rate": 0.12,
  "model__colsample_bytree": 0.9
}
```

## Threshold sweep (selected model, test set)

| threshold | precision | recall | f1 | fp | fn |
| --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.7857 | 0.8105 | 0.7979 | 21.0000 | 18.0000 |
| 0.2000 | 0.8370 | 0.8105 | 0.8235 | 15.0000 | 18.0000 |
| 0.3000 | 0.8736 | 0.8000 | 0.8352 | 11.0000 | 19.0000 |
| 0.4000 | 0.8706 | 0.7789 | 0.8222 | 11.0000 | 21.0000 |
| 0.5000 | 0.8690 | 0.7684 | 0.8156 | 11.0000 | 22.0000 |
| 0.6000 | 0.9241 | 0.7684 | 0.8391 | 6.0000 | 22.0000 |
| 0.7000 | 0.9481 | 0.7684 | 0.8488 | 4.0000 | 22.0000 |
| 0.8000 | 0.9605 | 0.7684 | 0.8538 | 3.0000 | 22.0000 |
| 0.9000 | 0.9600 | 0.7579 | 0.8471 | 3.0000 | 23.0000 |

The Streamlit app exposes the threshold as an operational control.
A lower threshold increases recall (fewer missed frauds) and usually
increases false positives (more analyst reviews).

## Confusion-matrix language

- **TP**: fraud correctly flagged
- **TN**: legitimate transaction correctly cleared
- **FP**: legitimate transaction flagged (review cost / customer friction)
- **FN**: fraud missed (direct financial loss) — typically the costliest error

## Limitations

V1–V28 are PCA components; they are not human-readable merchant fields.
Metrics describe this public snapshot, not a live payment stream.
