import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from src.features.feature_engineering import make_features
import mlflow
import mlflow.lightgbm

SEED = 42
TARGET = "target"

def main():
    with mlflow.start_run():
        mlflow.lightgbm.autolog()

        df = pd.read_parquet("input/train.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df = make_features(df).dropna()
        features = df.drop(columns=[TARGET, "date"]).columns
        X, y = df[features], df[TARGET]

        tscv = TimeSeriesSplit(n_splits=5)
        oof, scores = np.zeros(len(y)), []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                random_state=SEED,
                subsample=0.8,
                colsample_bytree=0.8
            )
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      eval_metric="rmse",
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
            oof[va_idx] = model.predict(X_va)
            score = mean_squared_error(y_va, oof[va_idx], squared=False)
            scores.append(score)
            print(f"Fold{fold}: {score:.4f}")

        cv_rmse = np.mean(scores)
        print(f"CV RMSE: {cv_rmse:.4f} ± {np.std(scores):.4f}")
        mlflow.log_metric("cv_rmse", cv_rmse)

if __name__ == "__main__":
    main()