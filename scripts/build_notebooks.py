"""Build educational Jupyter notebooks for FraudGuard."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


def _src(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def write(name: str, cells: list[dict]) -> None:
    path = NB / name
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print("wrote", path)


BOOT = """
from pathlib import Path
import sys
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_processing import dataset_overview, drop_duplicates_if_present, load_raw_dataset
from src.utils import RAW_DATASET_PATH
"""


def nb01() -> None:
    write(
        "01_data_understanding.ipynb",
        [
            md(
                """# 01 — Data understanding

**Why this notebook exists:** before modeling, we need to know *what* we have, *how big* it is, *what is missing*, and *how rare fraud is*. Skipping this step is how leakage, silent duplicates, and misleading accuracy sneak into a project.

## Dataset source

We use the **Credit Card Fraud Detection** dataset published by the [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Université Libre de Bruxelles). A public CSV mirror (the same file used in TensorFlow tutorials) is downloaded by `python scripts/download_data.py`.

Transactions were collected over two days. Features **V1–V28** are **PCA components** of confidential raw attributes. **Time** and **Amount** were not transformed. **Class** is the target: `1` = fraud, `0` = legitimate.

## Target variable

This is a **binary classification** problem: assign each transaction to one of two classes.

1. **What it is:** learning a mapping from features → class (or class probability).
2. **Why we need it:** reviewers need a score, not an unsupervised cluster id.
3. **How it works:** a model estimates \(P(\\text{fraud} \\mid x)\) and a threshold turns that into a yes/no flag.
4. **In this project:** `Class` is that label. It is extremely imbalanced.

## Limitations (read this twice)

- PCA features are **not interpretable merchant fields**.
- Two days of European card traffic ≠ your bank's 2026 stream (**distribution shift**).
- Labels can be delayed or incomplete in real life; here they are given as ground truth.
- We will **not fabricate** row counts. Run the cells on the real file.
"""
            ),
            code(BOOT),
            md(
                """## Load the file

**Why:** if the CSV is missing, every later notebook is theater. Fail loudly with a path the user can fix.
"""
            ),
            code(
                """
from pathlib import Path
print("Expected CSV:", RAW_DATASET_PATH)
print("Exists:", RAW_DATASET_PATH.exists())
df = load_raw_dataset()
df.head()
"""
            ),
            md(
                """## Shape, columns, dtypes

**Why:** shape tells us whether we can afford heavy models; dtypes tell us what must be encoded or scaled; column names tell us what we are *not* allowed to treat as causal (the `V*` fields).
"""
            ),
            code(
                """
print("shape:", df.shape)
print("columns:", list(df.columns))
df.dtypes
"""
            ),
            md(
                """## Descriptive statistics

**Why:** `Amount` is a heavy-tailed money field; `Time` is a clock in seconds from the first event; `V*` should look roughly centered because they are PCA scores. We are looking for impossible negatives on Amount, empty columns, or surprise non-numeric types — not for a story we invented before seeing the data.
"""
            ),
            code("df.describe().T"),
            md(
                """## Missing values

**Why:** sklearn estimators generally refuse `NaN`. If missingness is related to fraud, dropping rows could bias the label rate. This public file is known to be complete — we still **measure** it instead of assuming.
"""
            ),
            code(
                """
missing = df.isna().sum()
print("total missing cells:", int(missing.sum()))
missing[missing > 0]
"""
            ),
            md(
                """## Duplicate records

**Why:** an identical row in both train and test is a form of **leakage**. The model "sees" the test transaction during training. Exact duplicates are not independent extra evidence.
"""
            ),
            code(
                """
n_dup = int(df.duplicated().sum())
print("duplicate rows:", n_dup)
print("duplicate rate:", n_dup / len(df))
"""
            ),
            md(
                """## Class distribution — the most important table in this project

**Why:** if 99.8% of rows are legitimate, **accuracy is the wrong headline metric**. A model that predicts "legitimate" every time would look almost perfect and catch **zero fraud**.
"""
            ),
            code(
                """
overview = dataset_overview(df)
overview
"""
            ),
            code(
                """
counts = df["Class"].value_counts()
print(counts)
print("legitimate %:", 100 * counts.get(0, 0) / len(df))
print("fraud %:", 100 * counts.get(1, 0) / len(df))
"""
            ),
            md(
                """## Accuracy trap (toy arithmetic)

Suppose 99.8% of transactions are legitimate and 0.2% are fraud.

A constant "not fraud" classifier:

- accuracy ≈ 0.998
- recall for fraud = 0
- precision for fraud is undefined or 0

**That is why later notebooks optimize PR-AUC / recall / precision, not accuracy.**

## What we take into preprocessing

- Drop exact duplicates (justified: copies, not new events).
- Stratified train/test split so both sides keep the tiny fraud rate.
- Scale numeric fields **after** the split, fit on train only.
"""
            ),
        ],
    )


def nb02() -> None:
    write(
        "02_exploratory_data_analysis.ipynb",
        [
            md(
                """# 02 — Exploratory data analysis

Every chart below answers a **question**. Decorative plots are omitted on purpose.

## Questions

1. Are fraudulent amounts clustered at extremes?
2. How severe is class imbalance in **counts** vs **percentages**?
3. Does the two-day `Time` clock show periods with a higher fraud *rate*?
4. Which PCA features even *look* separable by class? (association, not causation)
"""
            ),
            code(
                BOOT
                + """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.feature_engineering import add_engineered_features

sns.set_theme(style="whitegrid", context="notebook")
df = add_engineered_features(drop_duplicates_if_present(load_raw_dataset()))
print(df.shape)
"""
            ),
            md(
                """## Class imbalance

**Question:** how many fraud cases do we actually have to learn from?

**Why:** tree ensembles can still work with hundreds of positives, but resampling and class weights only make sense once we have seen the counts.
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
labels = df["Class"].map({0: "Legitimate", 1: "Fraud"})
labels.value_counts().plot(kind="bar", ax=axes[0], color=["#2c5364", "#c45c26"])
axes[0].set_title("Absolute counts")
axes[0].set_ylabel("Transactions")
(labels.value_counts(normalize=True) * 100).plot(kind="bar", ax=axes[1], color=["#2c5364", "#c45c26"])
axes[1].set_title("Percent of dataset")
axes[1].set_ylabel("%")
fig.tight_layout()
"""
            ),
            md(
                """## Amount distribution

**Question:** are fraudulent transactions concentrated around particular amounts?

**Why:** a rule like "flag Amount > 1000" is tempting and often wrong. Fraudsters frequently test small charges.
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.kdeplot(data=df, x="Amount", hue="Class", common_norm=False, clip=(0, 500), ax=axes[0])
axes[0].set_title("Amount KDE (clipped at 500)")
sns.boxplot(data=df, x="Class", y="Amount", showfliers=False, ax=axes[1])
axes[1].set_title("Amount by class (outliers hidden)")
fig.tight_layout()
df.groupby("Class")["Amount"].describe()
"""
            ),
            md(
                """## Time / hour-of-day proxy

**Question:** does fraud rate vary over the 24-hour wrap of `Time`?

`Time` is seconds since the **first transaction in the file**, not a calendar datetime. `(Time % 86400) / 3600` is a convenient daily clock, not "14:00 in Brussels".
"""
            ),
            code(
                """
hourly = df.groupby(df["hour_of_day"].astype(int)).agg(
    n=("Class", "size"),
    fraud_rate=("Class", "mean"),
    mean_amount=("Amount", "mean"),
)
fig, ax = plt.subplots(figsize=(8, 4))
hourly["fraud_rate"].plot(ax=ax, color="#c45c26")
ax.set_xlabel("Hour-of-day proxy")
ax.set_ylabel("Fraud rate")
ax.set_title("Fraud rate by hour-of-day proxy")
hourly.head()
"""
            ),
            md(
                """## Feature analysis (PCA components)

**Question:** which `V*` distributions differ by class enough to be useful to a model?

We plot a few components that literature and prior runs often highlight (V4, V10, V12, V14, V17). **Different densities ≠ the feature caused fraud.** They are rotated mixtures of hidden raw fields.
"""
            ),
            code(
                """
cols = ["V4", "V10", "V12", "V14", "V17"]
fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharey=False)
for ax, col in zip(axes, cols):
    sns.kdeplot(data=df, x=col, hue="Class", common_norm=False, ax=ax, legend=False)
    ax.set_title(col)
fig.tight_layout()
"""
            ),
            md(
                """## Correlation snapshot

**Question:** are Amount, hour, and a handful of PCA features linearly associated with `Class`?

Pearson correlation with a 0.17% minority class is a blunt instrument. Weak correlation does **not** mean a nonlinear model cannot use the feature. Strong correlation still is **not causation**.
"""
            ),
            code(
                """
corr_cols = cols + ["Amount", "hour_of_day", "Class"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Pearson correlation (selected columns)")
fig.tight_layout()
"""
            ),
            md(
                """## Scatter (sampled)

**Question:** do V14 vs V12 show a visible fraud cloud, or is overlap huge?

Huge overlap is expected: that is why we need a multivariate model and why false positives will exist.
"""
            ),
            code(
                """
sample = pd.concat([
    df[df.Class == 0].sample(4000, random_state=42),
    df[df.Class == 1],
], ignore_index=True)
fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(data=sample, x="V12", y="V14", hue="Class", alpha=0.35, ax=ax, palette={0: "#2c5364", 1: "#c45c26"})
ax.set_title("V12 vs V14 (all fraud + 4000 legit)")
"""
            ),
        ],
    )


def nb03() -> None:
    write(
        "03_data_preprocessing.ipynb",
        [
            md(
                """# 03 — Preprocessing (and data leakage)

## Data leakage

1. **What it is:** using information at training time that would not be available — or would be computed using the test distribution — in a fair evaluation.
2. **Why we care:** leaked models look brilliant in a notebook and fail quietly in production (or on a true holdout).
3. **How it happens here:** scaling on the full dataset; SMOTE before splitting; dropping rows using test labels; using `Class` as a feature.
4. **In this project:** `StandardScaler` lives inside a `Pipeline` and is **fit on training folds only**. Resampling happens **after** the split, **inside** training.

```
Raw rows
   ↓
Drop exact duplicates
   ↓
Deterministic features (hour, log amount)  ← no learned statistics
   ↓
Stratified train / test split
   ↓
Fit scaler + model on TRAIN
   ↓
Transform test with the *train* scaler
```
"""
            ),
            code(
                BOOT
                + """
from src.data_processing import build_model_pipeline, stratified_train_test_split
from src.feature_engineering import MODEL_FEATURE_COLUMNS
from sklearn.linear_model import LogisticRegression

df = drop_duplicates_if_present(load_raw_dataset())
X_train, X_test, y_train, y_test = stratified_train_test_split(df)
print(X_train.shape, X_test.shape)
print("train fraud rate:", float(y_train.mean()))
print("test fraud rate:", float(y_test.mean()))
print("features:", MODEL_FEATURE_COLUMNS)
"""
            ),
            md(
                """## Stratification

**What:** split so that class proportions in train and test match the original.

**Why ordinary random split is risky:** with ~500 fraud rows, an unlucky split can put almost all fraud in train or test. Metrics then swing wildly.

**How:** `train_test_split(..., stratify=y)`.

## Scaling

Linear models and k-NN (inside SMOTE) care about feature scale. Trees care less, but a **shared pipeline** keeps evaluation honest.

We still scale trees so SMOTE / logistic regression comparisons are fair.
"""
            ),
            code(
                """
pipe = build_model_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
pipe
"""
            ),
            md(
                """## Missing values and duplicates

This file has no missing cells (verify in notebook 01). Duplicates are dropped **before** splitting so the same fingerprint cannot appear on both sides.

If you ever impute, fit the imputer on **train only**, same as the scaler.
"""
            ),
            code(
                """
print("NaNs in train:", int(X_train.isna().sum().sum()))
print("NaNs in test:", int(X_test.isna().sum().sum()))
"""
            ),
        ],
    )


def nb04() -> None:
    write(
        "04_model_training.ipynb",
        [
            md(
                """# 04 — Training: baseline, imbalance, models, CV, tuning

This notebook **explains** the training story. The reproducible full run is:

```bash
python -m src.train
```

that writes `models/model.pkl`, `models/model_metadata.json`, and `reports/model_report.md`.

## Classification recap

Input features → model → probability → threshold → {0,1}.

## Logistic regression

```
features x
    ↓
weighted sum  z = w·x + b
    ↓
sigmoid  σ(z) = 1 / (1 + e^{-z})
    ↓
probability of fraud
    ↓
threshold (default 0.5, later tuned as policy)
    ↓
prediction
```

1. **What:** a linear classifier in probability space.
2. **Why:** strong, inspectable **baseline**. If logistic regression already separates well, trees must beat it on **PR-AUC**, not vibes.
3. **How:** maximum likelihood for Bernoulli labels.
4. **Here:** we compare unweighted vs `class_weight="balanced"`.

## Decision tree

Recursive splits that isolate purer regions of fraud vs legit.

## Random forest

```
dataset
    ↓
many trees on bootstrap samples / feature subsets
    ↓
each tree votes (or averages probabilities)
    ↓
final prediction
```

## Gradient boosting / XGBoost

Trees are added **sequentially** to correct residual errors. XGBoost is a regularized, efficient implementation. `scale_pos_weight` ≈ negatives/positives addresses imbalance without synthesizing rows.

## Class imbalance methods

- **Class weights:** penalize mistakes on fraud more. No synthetic data.
- **Undersampling:** throw away legit rows. Fast, but wastes majority information.
- **SMOTE:**

```
minority class
    ↓
find nearby minority neighbors
    ↓
create synthetic points on the joining lines
    ↓
more balanced *training* data
```

**Never SMOTE before train/test split:** synthetic neighbors can be built from points that belong in the test set, leaking holdout geometry.

SMOTE also belongs **inside** cross-validation folds, not on the whole train set before CV (otherwise validation folds are not independent of the sampler).
"""
            ),
            code(
                BOOT
                + """
import pandas as pd
from pathlib import Path
reports = ROOT / "reports"
for name in ["imbalance_comparison.csv", "cv_model_comparison.csv", "test_metrics.csv"]:
    path = reports / name
    print("\\n===", name, "===")
    if path.exists():
        display(pd.read_csv(path))
    else:
        print("Run `python -m src.train` to generate this table.")
"""
            ),
            md(
                """## Cross-validation: StratifiedKFold

1. **What:** rotate which slice of *training* data is the validation fold.
2. **Why ordinary K-Fold fails here:** a fold might contain almost no fraud; ROC/PR scores become noisy or undefined.
3. **What stratification does:** each fold keeps roughly the same fraud rate.
4. **Why CV is more reliable than one split:** you see mean ± std of PR-AUC before touching the test set.

Hyperparameter search in `src/train.py` uses `RandomizedSearchCV` with `scoring="average_precision"` (PR-AUC) on **train** only.

## Parameters we search (and why)

**Random Forest / trees**

- `n_estimators`: more trees → more stable, slower.
- `max_depth`: deeper → more interaction capture, more overfit risk.
- `min_samples_split` / `min_samples_leaf`: stop tiny fraudulent leaves that do not generalize.
- `max_features`: randomness that decorrelates trees.

**XGBoost**

- `learning_rate`: smaller steps need more trees but often generalize better.
- `subsample` / `colsample_bytree`: row/column dropout against overfit.
- `min_child_weight`: ignore ultra-small child nodes.

Search spaces stay small on purpose: a portfolio project should finish on a laptop.
"""
            ),
            code(
                """
import json
meta_path = ROOT / "models" / "model_metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print("Selected model:", meta.get("model_name"))
    print("Threshold:", meta.get("threshold"))
    print("Best params:", meta.get("best_params"))
else:
    print("No metadata yet — train first.")
"""
            ),
        ],
    )


def nb05() -> None:
    write(
        "05_model_evaluation.ipynb",
        [
            md(
                """# 05 — Evaluation, threshold analysis, risk scores

## Confusion matrix language

| | Predicted legit | Predicted fraud |
|---|---|---|
| **Actual legit** | TN | FP |
| **Actual fraud** | FN | TP |

- **TP** — fraud we caught.
- **TN** — legitimate traffic we left alone.
- **FP** — false alarm (analyst time, blocked customer).
- **FN** — missed fraud (**usually the most expensive** in payments: the money is already gone).

We do **not** pick a "best" model from accuracy.

## Precision

Of rows we flagged as fraud, what fraction were truly fraud?

High precision → fewer wasted reviews. Too high a precision target often means we hide in conservative thresholds and miss fraud.

## Recall

Of *all* actual fraud, what fraction did we catch?

High recall → fewer FN. The cost is more FP.

## F1

Harmonic mean of precision and recall. Useful when you want a single compromise, not when one error type is 100× more expensive than the other.

## ROC-AUC

Area under TPR vs FPR. Can look optimistic on rare events because many TN make FPR tiny.

## PR-AUC (average precision)

Area under precision vs recall. The **no-skill baseline is the fraud prevalence** (~0.002), not 0.5. That is why PR-AUC is the headline ranking metric in this repo.

## Thresholds

`0.5` is a convenience, not a business law. Lower threshold → more recall, more FP. The app exposes this as policy.
"""
            ),
            code(
                BOOT
                + """
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluate import classification_metrics, threshold_sweep

payload_path = ROOT / "models" / "eval_payload.json"
meta_path = ROOT / "models" / "model_metadata.json"
assert payload_path.exists(), "Run python -m src.train first"
payload = json.loads(payload_path.read_text())
y_true = np.array(payload["y_true"])
y_prob = np.array(payload["y_prob"])
metrics = classification_metrics(y_true, y_prob, threshold=0.5)
metrics
"""
            ),
            code(
                """
sweep = threshold_sweep(y_true, y_prob)
sweep
"""
            ),
            code(
                """
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sweep.threshold, sweep.precision, marker="o", label="precision")
ax.plot(sweep.threshold, sweep.recall, marker="o", label="recall")
ax.plot(sweep.threshold, sweep.f1, marker="o", label="f1")
ax.set_xlabel("threshold")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Threshold policy curve")
"""
            ),
            md(
                """## Risk score vs probability vs category

- **Model probability:** `predict_proba` output in \\([0,1]\\). Not automatically a calibrated long-run frequency.
- **Risk score:** `round(100 * probability)` — a communication scale.
- **Category:** LOW (0–30), MEDIUM (31–70), HIGH (71–100) — **policy bands**.

`predict_risk(transaction)` in `src/predict.py` returns all four of: probability, score, category, binary prediction.
"""
            ),
            code(
                """
from src.predict import probability_to_risk_score, risk_category
for p in [0.05, 0.4, 0.87]:
    s = probability_to_risk_score(p)
    print(p, s, risk_category(s))
"""
            ),
            md(
                """Open `reports/model_report.md` for the full comparison table produced on your machine. Never paste invented metrics into a CV.
"""
            ),
        ],
    )


def nb06() -> None:
    write(
        "06_model_explainability.ipynb",
        [
            md(
                """# 06 — Explainability (SHAP and honest caveats)

## Feature importance vs SHAP

**Tree feature importance** (gain/weight) is **global** and can be biased toward high-cardinality splits.

**SHAP** (SHapley Additive exPlanations):

1. **What:** attributes a prediction to features using a game-theoretic average of marginal contributions.
2. **Why:** reviewers ask "why did *this* payment look like fraud?"
3. **How:** TreeSHAP computes exact Shapley values for tree ensembles efficiently.
4. **Here:** global summary on a sample + one local explanation.

## Hard rule

SHAP tells you **what the model used**. It does **not** prove that a PCA component *caused* fraud. Correlation is not causation; attribution is not causation either.

V1–V28 are anonymized. Saying "V14 caused the decline" would be scientifically false.
"""
            ),
            code(
                BOOT
                + """
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.predict import load_artifacts, predict_risk
from src.data_processing import stratified_train_test_split

artifacts = load_artifacts()
meta = artifacts["metadata"]
print("model:", meta.get("model_name"))
df = drop_duplicates_if_present(load_raw_dataset())
X_train, X_test, y_train, y_test = stratified_train_test_split(df)
"""
            ),
            md("## Local explanation for one test row (app-style contributions)")
            ,
            code(
                """
row = X_test.iloc[0]
result = predict_risk(row, artifacts=artifacts)
print("probability", result["fraud_probability"])
print("score", result["risk_score"], result["risk_category"])
pd.DataFrame(result["contributing_features"])
"""
            ),
            md(
                """## Global SHAP (sample)

If this cell fails (install issues), the application still uses the contribution fallback in `predict.py`. That is intentional: the product should not hard-crash without SHAP.
"""
            ),
            code(
                """
try:
    import shap
    pipe = artifacts["pipeline"]
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocess"]
    sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
    Xt = pre.transform(sample)
    names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    shap.summary_plot(sv, Xt, feature_names=names, show=True)
except Exception as exc:
    print("SHAP skipped:", type(exc).__name__, exc)
"""
            ),
        ],
    )


def main() -> None:
    NB.mkdir(parents=True, exist_ok=True)
    nb01()
    nb02()
    nb03()
    nb04()
    nb05()
    nb06()


if __name__ == "__main__":
    main()
