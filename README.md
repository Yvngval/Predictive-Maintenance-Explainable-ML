# Predictive Maintenance & Explainable Machine Learning

Predicting industrial equipment failures from sensor data, and explaining *why* the model predicts them — not just *that* it does.

## Problem

An industrial company wants to anticipate equipment failures to reduce production downtime and maintenance costs. It has operational sensor data from its machines, along with a label indicating whether a failure occurred.

**Goal:** build a Machine Learning model that predicts the occurrence of a failure, then analyze which factors drive its predictions.

**Research question:** To what extent can Machine Learning models effectively detect failure risk, and which factors most influence their predictions?

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 synthetic records of industrial machine operating conditions (air/process temperature, rotational speed, torque, tool wear) with a binary machine-failure label. Compact enough to iterate quickly, rich enough to support a real Data Science workflow.

## Method

1. **EDA** — variable distributions, correlations, failure rate, outliers, class imbalance.
2. **Preprocessing** — minimal cleaning, encoding where needed, train/test split, standardization for models that require it.
3. **Modeling**
   - Baseline: Logistic Regression
   - Random Forest
   - XGBoost
4. **Evaluation** — Precision, Recall, F1-score, PR-AUC and confusion matrix (accuracy is not a reliable metric under class imbalance and is reported for context only).
5. **Explainability** — SHAP applied to the best-performing model to identify which features drive individual and global predictions.
6. **Error analysis** — manual review of a sample of false positives and false negatives to understand failure modes.

### R&D angle

**Hypothesis:** non-linear models (Random Forest, XGBoost) should detect failures more effectively than logistic regression, but may produce more false positives.

This is tested experimentally by comparing the three models on the same splits and metrics, then explaining *why* the stronger model performs better using SHAP, and discussing the limits of the dataset (synthetic data, class imbalance, limited failure modes) and of the protocol.

## Results

*(fill in after running the experiments)*

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |
| XGBoost | | | | |

- Confusion matrix — best model
- PR curve — model comparison
- SHAP summary plot — global feature importance
- Notes from error analysis (false positives / false negatives)

## Key findings

*(fill in: which model performed best, which features SHAP identified as most influential, what the false positives/negatives had in common, and the main limitations of the dataset/protocol)*

## Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib

## Repository structure

```
.
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── data/                 # raw / processed data (or a download script, if the dataset can't be committed)
├── figures/              # confusion matrix, PR curve, SHAP summary plot
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_eda.ipynb
```

## Limitations

- Dataset is synthetic (AI4I 2020), so patterns may not fully generalize to real industrial sensor data.
- Failure classes are highly imbalanced; results should be read alongside PR-AUC and Recall, not accuracy.
- Model scope is intentionally limited to 3 classical ML models — no deep learning, no deployment — to keep the project focused on the Data Science methodology (EDA → modeling → evaluation → interpretation) rather than on infrastructure.
