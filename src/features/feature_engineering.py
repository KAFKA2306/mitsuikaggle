import pandas as pd

TARGET = "target" # この変数は make_features 関数内で使用されるため、feature_engineering.py にも定義が必要です。

def make_features(df: pd.DataFrame, lags=[1,7,30], windows=[7,30]):
    df = df.sort_values("date")
    for lag in lags:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df[TARGET].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df[TARGET].shift(1).rolling(w).std()
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    return df