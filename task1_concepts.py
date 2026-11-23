import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from common import OUT,RAW_NUM,load_data,split_data,TARGET
df=load_data()
x_train,x_val,x_test,y_train,y_val,y_test=split_data(df)
print(f"split sizes train={len(x_test)} (test set put aside)")
print(f"default rate train={y_train.mean():.3f} val={y_val.mean():.3f}test={y_test.mean():.3f}\n")
# Simple numeric-only view for the concept demos (full preprocessing comes in Task 2)
num = [c for c in RAW_NUM if c != "loan_amount"] + ["loan_amount"]
imp = SimpleImputer(strategy="median").fit(x_train[num])
Xtr, Xva = imp.transform(x_train[num]), imp.transform(x_val[num])

# --- 1. Overfitting: tree depth sweep ------------------------------------------------
print("== Overfitting: DecisionTreeClassifier depth sweep ==")
depths = [1, 3, 5, 10, None]
tr_acc, va_acc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xtr, y_train)
    tr_acc.append(t.score(Xtr, y_train)); va_acc.append(t.score(Xva, y_val))
    print(f"Depth {str(d):>4}: train {tr_acc[-1]:.2f} val {va_acc[-1]:.2f}")
labels = [str(d) for d in depths]
plt.figure(figsize=(6, 4))
plt.plot(labels, tr_acc, "o-", label="train"); plt.plot(labels, va_acc, "s-", label="validation")
plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.title("Overfitting: train vs validation")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/task1_overfitting.png", dpi=130); plt.close()

# --- 2. 5-fold CV --------------------------------------------------------------------
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                   LogisticRegression(max_iter=1000))
cv = cross_val_score(lr, x_train[num], y_train, cv=5, scoring="accuracy")
print(f"\nCV (5-fold, LogisticRegression): {cv.mean():.2f} +/- {cv.std():.2f}")
print(f"  per fold: {np.round(cv, 3)}")

# --- 3. Data leakage -----------------------------------------------------------------
print("\n== Data leakage demo ==")

Xall = SimpleImputer(strategy="median").fit_transform(df[num])
yall = df[TARGET]

# WRONG: scaler fitted on ALL rows (incl. rows that end up in validation) before splitting
Xs = StandardScaler().fit_transform(Xall)
a_tr, a_va, b_tr, b_va = train_test_split(Xs, yall, test_size=0.25, stratify=yall, random_state=42)
wrong = LogisticRegression(max_iter=1000).fit(a_tr, b_tr).score(a_va, b_va)

# RIGHT: split first, fit scaler on train only, then transform validation with train statistics
a_tr, a_va, b_tr, b_va = train_test_split(Xall, yall, test_size=0.25, stratify=yall, random_state=42)
sc = StandardScaler().fit(a_tr)
right = LogisticRegression(max_iter=1000).fit(sc.transform(a_tr), b_tr).score(sc.transform(a_va), b_va)
print(f"Scale-before-split (leaky)   val accuracy: {wrong:.4f}")
print(f"Scale-after-split  (correct) val accuracy: {right:.4f}")
print(f"Scaler mean of 'income': leaky={StandardScaler().fit(Xall).mean_[1]:.1f}  "
      f"train-only={sc.mean_[1]:.1f}")
leak_text = """Explanation (4 lines):
1. Scaling before the split computes mean/std using validation rows too, so information about unseen data leaks into training.
2. The correct way fits the scaler on the training set only and merely applies (transform) those statistics to validation/test.
3. With a simple scaler and logistic regression the score difference is tiny, but it is a real bias and grows with
   target-aware steps (imputation, feature selection, target encoding) or small datasets.
4. Putting all preprocessing inside a Pipeline makes this automatic, because it is re-fitted on the training fold in every CV split."""
print(leak_text)

# --- 4. Problem types + workflow -----------------------------------------------------
concepts = """
== Supervised vs unsupervised ==
Spam detection          -> Supervised (classification: labelled spam / not spam emails)
Customer segmentation   -> Unsupervised (clustering: no labels, find groups, e.g. K-Means)
House price prediction  -> Supervised (regression: numeric target = price)
Anomaly detection       -> Unsupervised (usually; learn 'normal' and flag outliers, e.g. Isolation Forest;
                           it becomes supervised if labelled fraud examples exist)

== ML workflow in 7 steps ==
1. Define the problem & success metric (e.g. predict default; optimise recall/ROC-AUC).
2. Collect and understand the data (EDA, target balance, missing values, leakage risks).
3. Split into train / validation / test first (stratified) and put the test set aside.
4. Preprocess & engineer features (impute, encode, scale) inside a Pipeline fitted on train only.
5. Train baseline and candidate models; compare with cross-validation on train/validation.
6. Tune hyper-parameters (GridSearchCV / RandomizedSearchCV), pick threshold, check over-fitting.
7. Evaluate the single best model once on the test set, save it (joblib) and monitor in production.
"""
print(concepts)
with open(f"{OUT}/task1_concepts.txt", "w") as f:
    f.write(leak_text + "\n" + concepts)
