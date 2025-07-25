# EDA プロット結果

## 1. 主要な金融商品の価格推移

主要な金融商品の価格推移を時系列でプロットしました。

### LME_AH_Close, JPX_Gold_Standard_Futures_Close, US_Stock_SPY_adj_close, FX_USDJPY の価格トレンド

![Price Trends](plots/price_trends.png)

## 2. 価格差系列の挙動

選択された価格差系列と、対応するターゲット変数の時系列プロットです。

### Target 1: LME_PB_Close - US_Stock_VT_adj_close (lag 1)

![Price Difference Target 1](plots/price_diff_target_1.png)

### Target 2: LME_CA_Close - LME_ZS_Close (lag 1)

![Price Difference Target 2](plots/price_diff_target_2.png)

## 3. 欠損値の分布

各データセットにおける欠損値の割合を示します。

### train.csv の欠損値

![Missing Values in Train](plots/missing_values_train.png)

### train_labels.csv の欠損値

![Missing Values in Train Labels](plots/missing_values_train_labels.png)