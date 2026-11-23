# Week 2 Assessment: Loan Default ML Workflow

## Run
```
pip install scikit-learn xgboost pandas matplotlib joblib tabulate scipy
python task1_concepts.py && python task2_pipeline.py && python task3_models.py && python task4_evaluation.py
```
Plots, tables and the saved pipeline go to `outputs/`.

## Dataset (important)
The trainer's dataset was not provided, so `common.py` generates a **synthetic, seeded bank-loan dataset**
(6,000 rows, ~21% default, ~5% injected missing values, skewed income, a date column, an ordered
`education` column). To use the real dataset, put it at `data/loans.csv` with the same columns
(or edit the column lists at the top of `common.py`); nothing else needs to change.
Because the data is synthetic, absolute scores are only illustrative.

## Files
| File | Task |
|---|---|
| `common.py` | data, 60/20/20 stratified split (random_state=42), feature engineering, ColumnTransformer/Pipeline builders |
| `task1_concepts.py` | splits, overfitting plot, 5-fold CV, leakage demo + 4-line explanation, problem types, 7-step workflow |
| `task2_pipeline.py` | SimpleImputer vs KNNImputer, OneHot/Ordinal encoding, 3+ new features, Pipeline CV, joblib save/load + 3 raw-row predictions |
| `task3_models.py` | 4 classifiers (RF OOB, XGBoost early stopping on validation), predict_proba, top-5 features, Linear/Ridge/Lasso, K-Means k=2..8 |
| `task4_evaluation.py` | metrics + confusion matrices, ROC chart, threshold for recall >= 75%, GridSearchCV (RF), RandomizedSearchCV (XGB, n_iter=15), regression vs baseline, single test evaluation |

## Key decisions
- **Split first, preprocess inside a Pipeline.** Imputing, encoding, scaling and feature creation all live
  inside `Pipeline`, so they are re-fitted on each training fold. No leakage, and raw rows can be scored directly.
- **Test set untouched** until the very end of Task 4 (evaluated once).
- **Features:** date decomposition (month, weekday, quarter), `income_to_loan` interaction, log transform of the skewed
  `income` and `loan_amount`.
- **Regression target = `loan_amount`**, so regression features exclude anything derived from it.
- **Accuracy is misleading** here: ~79% of applicants do not default, so "always predict no default" is ~79% accurate with 0 recall.
  Models are compared on ROC-AUC, and the threshold is lowered to reach 75% recall.
- Lasso uses a strong alpha (300) on standardised features, which removes 17 of 21 features, keeping income, credit score, income x credit and education.

## Results (validation unless noted)
| Model | Accuracy | ROC-AUC |
|---|---|---|
| LogisticRegression | 0.798 | 0.681 |
| DecisionTree | 0.794 | 0.659 |
| RandomForest | 0.797 | 0.644 |
| XGBoost (early stopping) | 0.797 | 0.675 |

- Pipeline CV accuracy: 0.80 (+/- 0.01). Predictions for 3 new rows: [0 0 0].
- Regression (loan_amount, validation): MAE 2,329, RMSE 3,260, R2 0.83 vs mean baseline MAE 5,914, RMSE 7,946, R2 0.00.
- Threshold for 75% recall: 0.179 (precision 0.284 on validation).
- **Final test (used once):** LogisticRegression at threshold 0.179: ROC-AUC 0.69, recall 0.735, precision 0.284, F1 0.41.
- Overfitting: depth 1 train 0.79 / val 0.79 -> depth None train 1.00 / val 0.71.
- 5-fold CV (logistic): 0.79 +/- 0.01. Pipeline CV accuracy: 0.79.

- K-Means: best silhouette at k=2 (0.43): a large lower-income/smaller-loan segment (~78%) and a smaller high-income/large-loan segment (~22%). Plot: `outputs/task3_elbow_silhouette.png`.
