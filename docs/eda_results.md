い## 探索的データ分析 (EDA) 結果

### 1. データセットのロードと確認

`input/train.csv` と `input/train_labels.csv` を正常にロードしました。

### 2. 各データセットの基本的な統計情報

#### `df_train` の基本情報

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1917 entries, 0 to 1916
Columns: 558 entries, date_id to FX_ZARGBP
dtypes: float64(557), int64(1)
memory usage: 8.2 MB
```

#### `df_train` の記述統計

```
           date_id  LME_AH_Close  ...    FX_NOKJPY    FX_ZARGBP
count  1917.000000   1867.000000  ...  1917.000000  1917.000000
mean    958.000000   2245.249839  ...    13.067047     0.048746
std     553.534552    400.328518  ...     0.959581     0.005288
min       0.000000   1462.000000  ...     9.618859     0.039464
25%     479.000000   1911.250000  ...    12.528221     0.043534
50%     958.000000   2236.500000  ...    13.231644     0.048767
75%    1437.000000   2496.000000  ...    13.781044     0.052576
max    1916.000000   3849.000000  ...    15.314668     0.062025

[8 rows x 558 columns]
```

#### `df_train_labels` の基本情報

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1917 entries, 0 to 1916
Columns: 425 entries, date_id to target_423
dtypes: float64(424), int64(1)
memory usage: 6.2 MB
```

#### `df_train_labels` の記述統計

```
           date_id     target_0  ...   target_422   target_423
count  1917.000000  1787.000000  ...  1699.000000  1728.000000
mean    958.000000     0.000438  ...     0.001256    -0.002853
std     553.534552     0.012029  ...     0.029830     0.054052
min       0.000000    -0.123763  ...    -0.222898    -0.212359
25%     479.000000    -0.004643  ...    -0.015038    -0.032515
50%     958.000000     0.000854  ...     0.000795    -0.001795
75%    1437.000000     0.006350  ...     0.017409     0.028380
max    1916.000000     0.087470  ...     0.130926     0.244952

[8 rows x 425 columns]
```

### 3. 欠損値の有無と数

#### `df_train` の欠損値

`df_train` には多数の欠損値が存在します。上位の欠損値を持つ列は以下の通りです。

```
LME_AH_Close                   50
LME_CA_Close                   50
LME_PB_Close                   50
LME_ZS_Close                   50
JPX_Gold_Mini_Futures_Open    115
...
US_Stock_X_adj_volume          64
US_Stock_XLB_adj_volume        64
US_Stock_XLE_adj_volume        64
US_Stock_XOM_adj_volume        64
US_Stock_YINN_adj_volume       64
Length: 519, dtype: int64
```

#### `df_train_labels` の欠損値

`df_train_labels` にも多数の欠損値が存在します。上位の欠損値を持つ列は以下の通りです。

```
target_0      130
target_1      173
target_2       86
target_3       86
target_4      286
...
target_419    102
target_420    338
target_421    189
target_422    218
target_423    189
Length: 422, dtype: int64
```

### 4. 主要な列のユニーク値の数と分布の概要

#### `df_train` のユニーク値の数

`date_id` は1917のユニーク値を持つため、日付または時系列インデックスとして機能していると考えられます。
その他の多くの列もユニーク値が多いため、数値データとして連続的な値を持っている可能性が高いです。

```
列 'date_id': 1917 ユニーク値 (多すぎるため分布は省略)
列 'LME_AH_Close': 1280 ユニーク値 (多すぎるため分布は省略)
列 'LME_CA_Close': 1633 ユニーク値 (多すぎるため分布は省略)
... (省略)
```

#### `df_train_labels` のユニーク値の数

`df_train_labels` の `target_` 列もほとんどがユニーク値が多いため、連続的なターゲット変数であると考えられます。

```
列 'target_0': 1787 ユニーク値 (多すぎるため分布は省略)
列 'target_1': 1744 ユニーク値 (多すぎるため分布は省略)
... (省略)