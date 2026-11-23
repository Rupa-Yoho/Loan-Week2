"""Task 2-preprocessing,feture and pipeline"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from common import OUT,add_features_clf,load_data,make_pipeline,split_data
x_train,x_val,x_test,y_train,y_val,y_test=split_data()
print("Missing values per column(train):")
print(x_train.isna().sum()
      [x_train.isna().sum()>0].to_string(),"\n")

#imputation:simpleimputer(median/most_frequency)vs KNNimputer on numeric columns

lr=LogisticRegression(max_iter=1000)
for name in ["median","knn"]:
    pipe=make_pipeline(lr,"clf",imputer=name)
    s=cross_val_score(pipe,x_train,y_train,cv=5,scoring="accuracy")
    print(f"numeric imputer={name:6s}:cv accuracy {s.mean():.4f}+/-{s.std():.4f}")

#--new features(3 engineering feature groups)
feats=add_features_clf(x_train.head(3))
print("\n Engineering features (first 3 rows):")
print(feats[["app_month","app_dayofweek","app_quarter",
             "income_to_loan","log_income","log_loan_amount"]].
             round(3).to_string())
print("\n Income skew before log:%.2f after log:%.2f"%(x_train["income"].
      skew(),np.log1p(x_train["income"]).skew()))

#full pipeline:features->column transformer(impute/encode/sclae)-->model
pipe=make_pipeline(LogisticRegression(max_iter=1000),"clf")
cv=cross_val_score(pipe,x_train,y_train,cv=5,scoring="accuracy")
print(f"\n Pipeline cv accuracy:{cv.mean():.2f} (+/-{cv.std():.2f})")
print("No preprocessing happens outside cv:imputers,encoders and scalar are re-fitted per fold.")

#save fitted pipeline,reload ,predict for 3 new raw rows
pipe.fit(x_train,y_train)
joblib.dump(pipe,f"{OUT}/loan_pipeline.joblib")
loaded=joblib.load(f"{OUT}/loan_pipeline.joblib")
new_rows=pd.DataFrame([dict(application_date="2026-03-15",age=29,income=32000,education="Bachelor",credit_score=540,
                            employment_years=2.0,num_open_accounts=9,
                            debt_to_income=0.62,home_ownership="Rent",loan_purpose="small_business",loan_amount=38000),
                        dict(application_date="2026-07-02",age=45,income=120000,education="Master",credit_score=780,employment_years=15.0,num_open_accounts=4,debt_to_income=0.18,
                             home_ownership="Own",loan_purpose="home_improvement",loan_amount=15000),
                             
                        dict(application_date="2026-09-21",age=38,income=np.nan,education=np.nan,credit_score=690,employment_years=np.nan,num_open_accounts=5,debt_to_income=0.35,
                                                     home_ownership="Mortage",loan_purpose="car",loan_amount=14000),
                                                     ])
print("\nPrediction for 3 new rows:",loaded.predict(new_rows))
print("P(default)for 3 new rows:",np.round(loaded.predict_proba
                                           (new_rows)[:,1],3))
new_rows.to_csv(f"{OUT}/new_rows.csv",index=False)

# ---------------- Show the encoders: OrdinalEncoder and OneHotEncoder ----------------
prep = pipe.named_steps["prep"]
ohe = prep.named_transformers_["cat"].named_steps["enc"] # OneHotEncoder
ordn = prep.named_transformers_["ord"].named_steps["enc"] # OrdinalEncoder
ohe_cols = list(ohe.get_feature_names_out(["home_ownership", "loan_purpose"]))

print("\n=== OrdinalEncoder (education: ordered column) ===")
print("Order used:", list(ordn.categories_[0]))
for i, c in enumerate(ordn.categories_[0]):
    print(f" {c} -> {i}")

print("\n=== OneHotEncoder (nominal columns) ===")
for col, cats in zip(["home_ownership", "loan_purpose"], ohe.categories_):
    print(f" {col}: {list(cats)}")
print("New columns:", ohe_cols)

sample = x_train.head(3)
print("\nBefore encoding:")
print(sample[["education", "home_ownership", "loan_purpose"]].to_string())

enc_cols = ["education"] + ohe_cols
after = prep.transform(add_features_clf(sample))
print("\nAfter encoding (education = number, others = 0/1):")
print(pd.DataFrame(after[:, -len(enc_cols):], columns=enc_cols).round(0).to_string(index=False))