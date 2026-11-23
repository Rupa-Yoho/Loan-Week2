"""Task 3 - Models: supervised and unsupervised."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score, silhouette_score
from sklearn.pipeline import make_pipeline as skl_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import OUT, feature_names, load_data, make_pipeline, split_data

X_train, X_val, X_test, y_train, y_val, y_test = split_data()
results = []

# ------------------------------------------------------------ 4 classifiers in pipelines
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1,
                                           random_state=42),
    "XGBoost": XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3,
                             early_stopping_rounds=30, eval_metric="auc", random_state=42),
}
fitted = {}
for name, model in models.items():
    pipe = make_pipeline(model, "clf")
    if name == "XGBoost":
        # early stopping on the VALIDATION set: fit preprocessing on train only, transform val
        pipe.named_steps["features"].fit(X_train)
        pre = pipe[:-1].fit(X_train) # fit features + prep on train only
        pipe.named_steps["model"].fit(pre.transform(X_train), y_train,
                                      eval_set=[(pre.transform(X_val), y_val)], verbose=False)
        print(f"XGBoost best iteration (early stopping): {pipe.named_steps['model'].best_iteration}")
    else:
        pipe.fit(X_train, y_train)
    fitted[name] = pipe
    proba = pipe.predict_proba(X_val)[:, 1]
    acc, auc = accuracy_score(y_val, pipe.predict(X_val)), roc_auc_score(y_val, proba)
    results.append((name, acc, auc))
    print(f"{name:20s} val accuracy {acc:.4f} val ROC-AUC {auc:.4f}")
print(f"RandomForest OOB score: {fitted['RandomForest'].named_steps['model'].oob_score_:.4f}\n")

# ------------------------------------------- predict_proba vs predict (logistic regression)
lr = fitted["LogisticRegression"]
sample = X_val.head(5)
proba = lr.predict_proba(sample)
print("LogisticRegression predict_proba vs predict (5 validation rows):")
print(pd.DataFrame({"P(no default)": proba[:, 0].round(3), "P(default)": proba[:, 1].round(3),
                    "predict": lr.predict(sample), "actual": y_val.head(5).values}).to_string())
print("predict() = class whose probability is >= 0.5; predict_proba gives the confidence behind it.\n")

# ----------------------------------------------------------- top-5 important features
names = feature_names(fitted["RandomForest"])
for name in ["RandomForest", "XGBoost"]:
    imp = fitted[name].named_steps["model"].feature_importances_
    top = pd.Series(imp, index=names).sort_values(ascending=False).head(5)
    print(f"Top 5 features - {name}:\n{top.round(4).to_string()}\n")

# --------------------------------------- Linear / Ridge / Lasso for loan_amount (regression)
yr_tr, yr_va = X_train["loan_amount"], X_val["loan_amount"]
reg_models = {"LinearRegression": LinearRegression(), "Ridge": Ridge(alpha=100.0),
              "Lasso": Lasso(alpha=300.0, max_iter=20000)}
coefs, reg_fit = {}, {}
print("Regression target: loan_amount")
for name, m in reg_models.items():
    p = make_pipeline(m, "reg").fit(X_train, yr_tr)
    reg_fit[name] = p
    coefs[name] = p.named_steps["model"].coef_
    r2 = r2_score(yr_va, p.predict(X_val)); mae = mean_absolute_error(yr_va, p.predict(X_val))
    results.append((name + " (loan_amount R2)", r2, np.nan))
    print(f"{name:17s} val R2 {r2:.4f} MAE {mae:,.0f}")
rnames = feature_names(reg_fit["Lasso"])
ctab = pd.DataFrame(coefs, index=rnames).round(1)
print("\nCoefficient comparison (standardised features):\n" + ctab.to_string())
nz = int((np.abs(coefs["Lasso"]) < 1e-9).sum())
print(f"\nLasso zero coefficients: {nz} of {len(rnames)} -> {[n for n, c in zip(rnames, coefs['Lasso']) if abs(c) < 1e-9]}")
print("Ridge shrinks coefficients towards 0 but keeps all; Lasso (L1) sets weak ones exactly to 0 = feature selection.\n")
ctab.to_csv(f"{OUT}/task3_coefficients.csv")

# ----------------------------------------------------------------------- K-Means (k=2..8)
seg_cols = ["income", "credit_score", "loan_amount"]
full = load_data()[seg_cols]
Z = skl_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(full)
ks, inertia, sil = list(range(2, 9)), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
    print(f"k={k}: inertia {inertia[-1]:,.0f} silhouette {sil[-1]:.3f}")
best_k = ks[int(np.argmax(sil))]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow curve", xlabel="k", ylabel="inertia")
ax[1].plot(ks, sil, "s-", color="tab:orange"); ax[1].axvline(best_k, ls="--", c="grey")
ax[1].set(title="Silhouette score", xlabel="k", ylabel="silhouette")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/task3_elbow_silhouette.png", dpi=130); plt.close()

km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Z)
seg = full.assign(segment=km.labels_)
profile = seg.groupby("segment").agg(customers=("income", "size"), income=("income", "mean"),
                                     credit_score=("credit_score", "mean"),
                                     loan_amount=("loan_amount", "mean")).round(0)
print(f"\nBest k by silhouette = {best_k}. Segment profiles (mean values):\n{profile.to_string()}")
for s, r in profile.iterrows():
    lvl = lambda v, col: "high" if v > full[col].mean() * 1.1 else ("low" if v < full[col].mean() * 0.9 else "mid")
    print(f" Segment {s}: {lvl(r.income,'income')} income, {lvl(r.credit_score,'credit_score')} "
          f"credit score, {lvl(r.loan_amount,'loan_amount')} loan amounts ({int(r.customers)} customers)")

pd.DataFrame(results, columns=["Model", "Validation score", "Validation ROC-AUC"]).to_csv(
    f"{OUT}/task3_results.csv", index=False)
print("\nResults table:\n" + pd.DataFrame(results, columns=["Model", "Validation score", "ROC-AUC"]).round(4).to_string(index=False))