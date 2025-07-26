import pandas as pd
import mlflow.lightgbm # モデルのロードに必要
from src.features.feature_engineering import make_features # make_features関数をインポート

TARGET = "target" # make_features関数で使用されるため、ここにも定義

def main():
    # テストデータの読み込み
    df_test = pd.read_csv("input/test.csv") # input/test.csv を読み込むことを想定
    df_test["date"] = pd.to_datetime(df_test["date"])

    # 特徴量エンジニアリング
    df_test = make_features(df_test).dropna() # make_featuresを適用

    features = df_test.drop(columns=["date"]).columns # TARGETカラムはテストデータにはないため除外

    # 学習済みモデルのロード (仮置き: 実際のRUN_IDに置き換える必要があります)
    # 例: model = mlflow.lightgbm.load_model("runs:/<YOUR_MLFLOW_RUN_ID>/model")
    # 現時点ではモデルがないため、ダミーの予測を生成します
    # 実際のモデルロードと予測ロジックは、モデルが学習されMLflowにログされた後に実装します。
    print("Warning: Model loading is currently a placeholder. Please replace with actual MLflow model loading.")
    # ダミーの予測値を生成 (実際の予測ロジックに置き換える)
    df_test["prediction"] = 0.0 # 仮の予測値

    # 提出ファイルの生成
    submission_df = df_test[["date", "prediction"]].copy()
    submission_df.rename(columns={"prediction": TARGET}, inplace=True) # TARGETカラム名に合わせる
    submission_df.to_csv("outputs/submission.csv", index=False)

    print("Submission file 'outputs/submission.csv' created successfully (with dummy predictions).")

if __name__ == "__main__":
    main()