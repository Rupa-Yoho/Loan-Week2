"""Load the saved pipeline and print what is inside it, plus predictions for the 3 new rows."""
import joblib
import pandas as pd
from common import OUT, feature_names

pipe = joblib.load(f"{OUT}/loan_pipeline.joblib")
rows = pd.read_csv(f"{OUT}/new_rows.csv")

lines = ["SAVED PIPELINE: outputs/loan_pipeline.joblib", "=" * 50, ""]
lines.append("Steps (in order):")
lines.append(" 1. features -> FunctionTransformer(add_features_clf): date parts, income_to_loan, log_income, log_loan_amount")
prep = pipe.named_steps["prep"]
for name, trans, cols in prep.transformers_:
    steps = " -> ".join(type(s).__name__ for _, s in trans.steps)
    lines.append(f" 2. prep/{name}: {steps} on {list(cols)}")
lines.append(f" 3. model -> {type(pipe.named_steps['model']).__name__}")
lines += ["", f"Final model input features ({len(feature_names(pipe))}):", " " + ", ".join(feature_names(pipe))]
lines += ["", "3 NEW RAW ROWS:", rows.to_string(index=False)]
proba = pipe.predict_proba(rows)[:, 1]
lines += ["", "PREDICTIONS (1 = default):"]
for i, (p, pr) in enumerate(zip(pipe.predict(rows), proba), 1):
    lines.append(f" Row {i}: predicted {p} (P(default) = {pr:.3f})")
text = "\n".join(lines)
print(text)
open(f"{OUT}/pipeline_summary.txt", "w").write(text)