"""Task 4 - Evaluation and tuning."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import OUT, make_pipeline, split_data

X_train, X_val, X_test, y_train, y_val, y_test = split_data()


def report(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[{name}] accuracy {accuracy_score(y_true, y_pred):.3f} | precision "
          f"{precision_score(y_true, y_pred, zero_division=0):.3f} | recall "
          f"{recall_score(y_true, y_pred):.3f} | F1 {f1_score(y_true, y_pred):.3f}")
    print(f" confusion matrix [[TN FP],[FN TP]] = {cm.tolist()}")


# ------------------------------------------------------- 1. metrics for every classifier
base = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                             eval_metric="auc", random_state=42),
}
print("=== Validation metrics (default threshold 0.5) ===")
fitted = {}
for n, m in base.items():
    fitted[n] = make_pipeline(m, "clf").fit(X_train, y_train)
    report(n, y_val, fitted[n].predict(X_val))
majority = max(y_val.mean(), 1 - y_val.mean())
print(f"\nWhy accuracy misleads: only {y_val.mean():.1%} of applicants default, so a model that "
      f"ALWAYS predicts 'no default' scores {majority:.1%} accuracy yet catches 0 defaulters "
      f"(recall 0). For imbalanced classes use recall, precision, F1 and ROC-AUC.")

# --------------------------------------------------------------- 2. hyper-parameter tuning
print("\n=== GridSearchCV: RandomForest (3 parameters, 3-fold CV, scoring=roc_auc) ===")
rf_grid = GridSearchCV(
    make_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1), "clf"),
    {"model__n_estimators": [200, 400], "model__max_depth": [4, 8, None],
     "model__min_samples_leaf": [1, 10, 30]},
    cv=3, scoring="roc_auc", n_jobs=1).fit(X_train, y_train)
print("Best params:", rf_grid.best_params_, "| CV ROC-AUC %.4f" % rf_grid.best_score_)

print("\n=== RandomizedSearchCV: XGBoost (n_iter=15, 3-fold CV, scoring=roc_auc) ===")
xgb_rand = RandomizedSearchCV(
    make_pipeline(XGBClassifier(eval_metric="auc", random_state=42), "clf"),
    {"model__n_estimators": randint(100, 500), "model__learning_rate": uniform(0.01, 0.15),
     "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4),
     "model__colsample_bytree": uniform(0.6, 0.4), "model__min_child_weight": randint(1, 10)},
    n_iter=15, cv=3, scoring="roc_auc", random_state=42, n_jobs=1).fit(X_train, y_train)
print("Best params:", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in xgb_rand.best_params_.items()}, "| CV ROC-AUC %.4f" % xgb_rand.best_score_)

tuned = {"LogisticRegression": fitted["LogisticRegression"],
         "RandomForest (tuned)": rf_grid.best_estimator_,
         "XGBoost (tuned)": xgb_rand.best_estimator_}
print("\n=== Tuned models on validation set ===")
val_auc = {}
for n, p in tuned.items():
    proba = p.predict_proba(X_val)[:, 1]
    val_auc[n] = roc_auc_score(y_val, proba)
    report(n, y_val, (proba >= 0.5).astype(int))
    print(f" ROC-AUC {val_auc[n]:.4f}")

# ------------------------------------------------------------------- 3. ROC curves chart
plt.figure(figsize=(6, 5))
for n, p in tuned.items():
    fpr, tpr, _ = roc_curve(y_val, p.predict_proba(X_val)[:, 1])
    plt.plot(fpr, tpr, label=f"{n} (AUC={val_auc[n]:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="chance"); plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)"); plt.title("ROC curves (validation set)")
plt.legend(loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/task4_roc_curves.png", dpi=130); plt.close()

# ------------------------------------------------- 4. threshold for recall >= 75% (on val)
best_name = max(val_auc, key=val_auc.get)
best = tuned[best_name]
proba_val = best.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, proba_val)
ok = np.where(rec[:-1] >= 0.75)[0]
i = ok[-1] # highest threshold that still gives recall >= 0.75
threshold = float(thr[i])
print(f"\nBest model on validation: {best_name} (ROC-AUC {val_auc[best_name]:.4f})")
print(f"Chosen threshold = {threshold:.3f} -> recall {rec[i]:.3f}, precision {prec[i]:.3f} (validation)")
print("Trade-off: lowering the threshold catches more defaulters but flags more good customers.")

# ------------------------------------------------------------ 5. regression vs baseline
print("\n=== Regression (loan_amount) vs mean baseline, validation set ===")
yr_tr, yr_va = X_train["loan_amount"], X_val["loan_amount"]
reg = make_pipeline(LinearRegression(), "reg").fit(X_train, yr_tr)
dummy = DummyRegressor(strategy="mean").fit(X_train, yr_tr)
rows = []
for n, m in [("LinearRegression", reg), ("Mean baseline", dummy)]:
    p = m.predict(X_val)
    rows.append((n, mean_absolute_error(yr_va, p), np.sqrt(mean_squared_error(yr_va, p)), r2_score(yr_va, p)))
rt = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).round(3)
print(rt.to_string(index=False))

# ----------------------------------------------- 6. FINAL: best model, test set ONCE
print("\n=== FINAL evaluation on the held-out TEST set (used once) ===")
proba_test = best.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= threshold).astype(int)
report(f"{best_name} @ threshold {threshold:.3f}", y_test, pred_test)
auc_test = roc_auc_score(y_test, proba_test)
f1_test = f1_score(y_test, pred_test)
print(f"Best model: {best_name} | test ROC-AUC {auc_test:.2f} | F1 {f1_test:.2f}")
print(f"Regression: MAE {rows[0][1]:,.0f} RMSE {rows[0][2]:,.0f} R2 {rows[0][3]:.2f} "
      f"vs baseline MAE {rows[1][1]:,.0f} RMSE {rows[1][2]:,.0f} R2 {rows[1][3]:.2f}")
pd.DataFrame([dict(model=best_name, threshold=threshold, test_auc=auc_test, test_f1=f1_test,
                   test_recall=recall_score(y_test, pred_test),
                   test_precision=precision_score(y_test, pred_test))]).to_csv(
    f"{OUT}/task4_final_test.csv", index=False)
rt.to_csv(f"{OUT}/task4_regression.csv", index=False)
