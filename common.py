import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer,OneHotEncoder,
                                   OrdinalEncoder,StandardScaler)
BASE=os.path.dirname(os.path.abspath(__file__))
DATA_PATH=os.path.join(BASE,"data","loans.csv")
OUT=os.path.join(BASE,"outputs")
os.makedirs(OUT,exist_ok=True)
TARGET="default"
EDU_ORDER=["High school","Bachelor","Master","Phd"]
CAT_COLS=["home_ownership","loan_purpose"]
ORD_COLS=["education"]
RAW_NUM=["age","income","credit_score","employment_years",
         "num_open_accounts","debt_to_income","loan_amount"]

###Data
def generate_data(n=6000, seed=42):
    rng=np.random.default_rng(seed)
    age=rng.integers(21,70,n)
    income=rng.lognormal(10.8,0.55,n).round(0)
    edu=rng.choice(EDU_ORDER,n,p=[0.3,0.4,0.22,0.08])
    edu_i=pd.Series(edu).map({k:i for i,k in enumerate(EDU_ORDER)}).values
    credit=np.clip(rng.normal(660,70,n)+15*edu_i,300,850).round(0)
    emp=np.clip((age-21)*rng.uniform(0.1,0.8,n),0,40).round(1)
    open_acc=rng.poisson(5,n)
    dti=np.clip(rng.normal(0.35,0.12,n),0.02,0.).round(3)
    home=rng.choice(["Rent","Own","Mortage"],n,p=[0.42,0.18,0.40])
    purpose=rng.choice(["debt_consolidation","home_improvement","car",
                        "small_business","education","medical"],
                        n,p=[0.35,0.2,0.15,0.1,0.1,0.1])
    start=pd.Timestamp("2022-01-01")
    dates=start+pd.to_timedelta(rng.integers(0,3*365,n),unit="D")
    loan=(0.22*income+18*(credit-600)+900*edu_i+2500*
          (purpose=="small_business")+
          rng.normal(0,2500,n))
    loan=np.clip(loan,1000,None).round(0)
    z=(4.0+(dti-0.35)-0.011*(credit-660)+1.6*(loan/income-0.22)-
       0.04*emp+0.35*(home=="RENT")+0.4*(purpose=="Small_business")+
       0.15*(purpose=="Medical")-0.05*edu_i+rng.normal(0,0.5,n))
    lo,hi=-10,10
#calibrate to -20% defaults
    for _ in range(50):
        mid=(lo+hi)/2
        if (1/(1+np.exp(-(z+mid)))).mean()>0.20:
            hi=mid
        else:
            lo=mid
    p=1/(1+np.exp(-(z+mid)))
    default=rng.binomial(1,p)
    df=pd.DataFrame(dict(application_date=dates.strftime("%Y-%m-%d"),age=age,income=income,education=edu,credit_score=credit,employment_years=emp,num_open_accounts=open_acc,debt_to_income=dti,home_ownership=home,
                         loan_purpose=purpose,loan_amount=loan,
                         default=default))
##injects missing values(MCAR)into several columns
    for col,frac in[("income",.06),("credit_score",.05),("employment_years",.08),("debt_to_income",.04),("education",.05),("home_ownership",.04),("loan_purpose",.03)]:
        df.loc[rng.random(n)<frac,col]=np.nan
    return df
def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH),exist_ok=True)
        generate_data().to_csv(DATA_PATH,index=False)
    return pd.read_csv(DATA_PATH)

def split_data(df=None):
    df=load_data()if df is None else df
    x,y=df.drop(columns=[TARGET]),df[TARGET]
    x_tmp,x_test,y_tmp,y_test=train_test_split(x,y,test_size=0.20,
                                               stratify=y,random_state=42)
    x_train,x_val,y_train,y_val=train_test_split(x_tmp,y_tmp,test_size=0.25,
                                                 stratify=y_tmp,random_state=42)
    return x_train,x_val,x_test,y_train,y_val,y_test
##feature engineering
def _date_parts(x):
    d=pd.to_datetime(x["application_date"])
    x["app_month"]=d.dt.month
    x["app_dayofweek"]=d.dt.dayofweek
    x["app_quarter"]=d.dt.quarter 
    return x

def add_features_clf(x):
    x=_date_parts(x.copy())
    x["income_to_loan"]=x["income"]/x["loan_amount"]
    x["log_income"]=np.log1p(x["income"])
    x["log_loan_amount"]=np.log1p(x["loan_amount"])
    return x.drop(columns=["application_date"])

def add_features_reg(X):
    """Regression target is loan_amount, so no feature may be derived 
    from it."""
    X = _date_parts(X.copy())
    X["log_income"] = np.log1p(X["income"])
    X["income_x_credit"] = X["income"] * X["credit_score"] / 1000
    return X.drop(columns=["application_date", "loan_amount"], errors="ignore")


NUM_CLF = [c for c in RAW_NUM] + ["app_month", "app_dayofweek", 
                                  "app_quarter",
                                  "income_to_loan", 
                                  "log_income", "log_loan_amount"]
NUM_REG = [c for c in RAW_NUM if c != "loan_amount"] + [
    "app_month", "app_dayofweek", "app_quarter", "log_income", 
    "income_x_credit"]


def build_preprocessor(num_cols, imputer="median", scale=True):
    num_steps = [("imp", KNNImputer(n_neighbors=5) if imputer == "knn"
                  else SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(categories=[EDU_ORDER]))]), ORD_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), CAT_COLS),
    ])


def make_pipeline(model, task="clf", imputer="median", scale=True):
    """FunctionTransformer(features) -> ColumnTransformer -> model, all in one Pipeline."""
    fe, cols = (add_features_clf, NUM_CLF) if task == "clf" else (add_features_reg, NUM_REG)
    return Pipeline([("features", FunctionTransformer(fe, validate=False)),
                     ("prep", build_preprocessor(cols, imputer, scale)),
                     ("model", model)])


def feature_names(pipe):
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]





