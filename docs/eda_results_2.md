# EDA2 結果

## 1. `target_pairs.csv`の分析

### 1.1. `input/target_pairs.csv`の基本的な情報

#### df.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 424 entries, 0 to 423
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   target  424 non-null    object
 1   lag     424 non-null    int64 
 2   pair    424 non-null    object
dtypes: int64(1), object(2)
memory usage: 10.1+ KB
```

#### df.head()
| target   |   lag | pair                                           |
|:---------|------:|:-----------------------------------------------|
| target_0 |     1 | US_Stock_VT_adj_close                          |
| target_1 |     1 | LME_PB_Close - US_Stock_VT_adj_close           |
| target_2 |     1 | LME_CA_Close - LME_ZS_Close                    |
| target_3 |     1 | LME_AH_Close - LME_ZS_Close                    |
| target_4 |     1 | LME_AH_Close - JPX_Gold_Standard_Futures_Close |

#### df.describe()
|       |       lag |
|:------|----------:|
| count | 424       |
| mean  |   2.5     |
| std   |   1.11935 |
| min   |   1       |
| 25%   |   1.75    |
| 50%   |   2.5     |
| 75%   |   3.25    |
| max   |   4       |

### 1.2. `lag`と`pair`列のユニーク値と分布

#### 列 `lag`
- ユニーク値の数: 4
- 分布:
|   lag |   count |
|------:|--------:|
|     1 |     106 |
|     2 |     106 |
|     3 |     106 |
|     4 |     106 |

#### 列 `pair`
- ユニーク値の数: 374
- 分布:
| pair                                                          |   count |
|:--------------------------------------------------------------|--------:|
| US_Stock_SCCO_adj_close - LME_AH_Close                        |       3 |
| FX_CADUSD - LME_CA_Close                                      |       2 |
| US_Stock_CVX_adj_close - LME_ZS_Close                         |       2 |
| FX_CADUSD - LME_AH_Close                                      |       2 |
| LME_ZS_Close - US_Stock_GLD_adj_close                         |       2 |
| FX_AUDCAD - JPX_Platinum_Standard_Futures_Close               |       2 |
| LME_ZS_Close - US_Stock_VYM_adj_close                         |       2 |
| FX_AUDJPY - LME_PB_Close                                      |       2 |
| LME_CA_Close - US_Stock_CCJ_adj_close                         |       2 |
| FX_AUDUSD - JPX_Platinum_Standard_Futures_Close               |       2 |
| US_Stock_RIO_adj_close - LME_PB_Close                         |       2 |
| LME_AH_Close - JPX_Platinum_Standard_Futures_Close            |       2 |
| FX_USDCHF - LME_PB_Close                                      |       2 |
| US_Stock_EWT_adj_close - LME_AH_Close                         |       2 |
| US_Stock_HL_adj_close - LME_AH_Close                          |       2 |
| FX_EURAUD - LME_AH_Close                                      |       2 |
| US_Stock_VGK_adj_close - LME_CA_Close                         |       2 |
| LME_AH_Close - US_Stock_OIH_adj_close                         |       2 |
| FX_NOKUSD - LME_AH_Close                                      |       2 |
| LME_PB_Close - US_Stock_SLB_adj_close                         |       2 |
| JPX_Gold_Standard_Futures_Close - US_Stock_VT_adj_close       |       2 |
| LME_PB_Close - US_Stock_STLD_adj_close                        |       2 |
| US_Stock_EOG_adj_close - LME_AH_Close                         |       2 |
| FX_GBPJPY - LME_CA_Close                                      |       2 |
| US_Stock_EEM_adj_close - LME_PB_Close                         |       2 |
| LME_AH_Close - US_Stock_VWO_adj_close                         |       2 |
| US_Stock_MS_adj_close - LME_PB_Close                          |       2 |
| LME_AH_Close - US_Stock_EFA_adj_close                         |       2 |
| US_Stock_DVN_adj_close - LME_ZS_Close                         |       2 |
| US_Stock_IEMG_adj_close - JPX_Platinum_Standard_Futures_Close |       2 |
| US_Stock_EWJ_adj_close - JPX_Gold_Standard_Futures_Close      |       2 |
| LME_ZS_Close - US_Stock_AMP_adj_close                         |       2 |
| US_Stock_URA_adj_close - JPX_Platinum_Standard_Futures_Close  |       2 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_EWY_adj_close  |       2 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_BKR_adj_close  |       2 |
| LME_AH_Close - US_Stock_BNDX_adj_close                        |       2 |
| LME_ZS_Close - US_Stock_IEF_adj_close                         |       2 |
| LME_ZS_Close - LME_PB_Close                                   |       2 |
| JPX_Gold_Standard_Futures_Close - LME_ZS_Close                |       2 |
| JPX_Gold_Standard_Futures_Close - FX_AUDCAD                   |       2 |
| FX_EURUSD - JPX_Gold_Standard_Futures_Close                   |       2 |
| JPX_Platinum_Standard_Futures_Close - FX_NZDCHF               |       2 |
| LME_AH_Close - US_Stock_NEM_adj_close                         |       2 |
| LME_AH_Close - US_Stock_XLE_adj_close                         |       2 |
| LME_CA_Close - US_Stock_RY_adj_close                          |       2 |
| LME_PB_Close - FX_NOKGBP                                      |       1 |
| LME_ZS_Close - US_Stock_OXY_adj_close                         |       1 |
| LME_ZS_Close - US_Stock_STLD_adj_close                        |       1 |
| LME_PB_Close - US_Stock_HES_adj_close                         |       1 |
| FX_ZARCHF - JPX_Gold_Standard_Futures_Close                   |       1 |
| US_Stock_XLE_adj_close - LME_PB_Close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_ALB_adj_close  |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_KMI_adj_close  |       1 |
| US_Stock_TECK_adj_close - LME_CA_Close                        |       1 |
| US_Stock_SPYV_adj_close - JPX_Platinum_Standard_Futures_Close |       1 |
| JPX_Gold_Standard_Futures_Close - FX_NOKGBP                   |       1 |
| LME_CA_Close - FX_NOKEUR                                      |       1 |
| US_Stock_DE_adj_close - LME_CA_Close                          |       1 |
| LME_CA_Close - US_Stock_EWZ_adj_close                         |       1 |
| US_Stock_SLB_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_AMP_adj_close      |       1 |
| US_Stock_CAT_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| US_Stock_BKR_adj_close - JPX_Platinum_Standard_Futures_Close  |       1 |
| LME_AH_Close - US_Stock_VEA_adj_close                         |       1 |
| FX_ZARJPY - LME_CA_Close                                      |       1 |
| US_Stock_OKE_adj_close - LME_ZS_Close                         |       1 |
| LME_AH_Close - US_Stock_EEM_adj_close                         |       1 |
| LME_PB_Close - FX_AUDUSD                                      |       1 |
| FX_NOKEUR - LME_PB_Close                                      |       1 |
| LME_PB_Close - FX_NOKJPY                                      |       1 |
| JPX_Platinum_Standard_Futures_Close - FX_EURGBP               |       1 |
| LME_ZS_Close - US_Stock_TD_adj_close                          |       1 |
| US_Stock_X_adj_close - LME_PB_Close                           |       1 |
| US_Stock_COP_adj_close - LME_PB_Close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_ACWI_adj_close |       1 |
| FX_CHFJPY - LME_ZS_Close                                      |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_SLV_adj_close  |       1 |
| US_Stock_CVX_adj_close - JPX_Platinum_Standard_Futures_Close  |       1 |
| US_Stock_CVE_adj_close - LME_AH_Close                         |       1 |
| FX_AUDJPY - LME_ZS_Close                                      |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_NUE_adj_close      |       1 |
| US_Stock_NEM_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| US_Stock_ENB_adj_close - LME_PB_Close                         |       1 |
| FX_EURJPY - JPX_Gold_Standard_Futures_Close                   |       1 |
| US_Stock_OKE_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| JPX_Gold_Standard_Futures_Close - LME_AH_Close                |       1 |
| LME_ZS_Close - FX_NOKUSD                                      |       1 |
| FX_ZARCHF - JPX_Platinum_Standard_Futures_Close               |       1 |
| JPX_Gold_Standard_Futures_Close - FX_ZARGBP                   |       1 |
| US_Stock_XLB_adj_close - LME_ZS_Close                         |       1 |
| FX_CADJPY - JPX_Platinum_Standard_Futures_Close               |       1 |
| LME_AH_Close - US_Stock_VGK_adj_close                         |       1 |
| US_Stock_MPC_adj_close - LME_CA_Close                         |       1 |
| LME_PB_Close - US_Stock_CLF_adj_close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_BNDX_adj_close |       1 |
| JPX_Platinum_Standard_Futures_Close - FX_ZARUSD               |       1 |
| LME_AH_Close - FX_GBPUSD                                      |       1 |
| LME_CA_Close - US_Stock_LYB_adj_close                         |       1 |
| FX_NZDJPY - LME_CA_Close                                      |       1 |
| LME_ZS_Close - US_Stock_CAT_adj_close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_DE_adj_close   |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_SPYV_adj_close     |       1 |
| US_Stock_EWJ_adj_close - LME_ZS_Close                         |       1 |
| LME_ZS_Close - FX_NOKGBP                                      |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_TRGP_adj_close |       1 |
| LME_PB_Close - FX_CHFJPY                                      |       1 |
| LME_ZS_Close - US_Stock_WPM_adj_close                         |       1 |
| US_Stock_GLD_adj_close - LME_PB_Close                         |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_RSP_adj_close      |       1 |
| US_Stock_HAL_adj_close - LME_CA_Close                         |       1 |
| LME_PB_Close - US_Stock_FCX_adj_close                         |       1 |
| US_Stock_VEA_adj_close - JPX_Platinum_Standard_Futures_Close  |       1 |
| FX_EURNZD - LME_PB_Close                                      |       1 |
| FX_ZARUSD                                                     |       1 |
| LME_PB_Close - FX_ZARUSD                                      |       1 |
| LME_CA_Close - JPX_Gold_Standard_Futures_Close                |       1 |
| LME_PB_Close - JPX_Gold_Standard_Futures_Close                |       1 |
| US_Stock_VXUS_adj_close - LME_CA_Close                        |       1 |
| LME_CA_Close - FX_EURUSD                                      |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_COP_adj_close      |       1 |
| LME_AH_Close - US_Stock_XLB_adj_close                         |       1 |
| FX_CADCHF - JPX_Platinum_Standard_Futures_Close               |       1 |
| US_Stock_NUE_adj_close - LME_CA_Close                         |       1 |
| US_Stock_ACWI_adj_close - JPX_Platinum_Standard_Futures_Close |       1 |
| LME_CA_Close - FX_NZDJPY                                      |       1 |
| US_Stock_OXY_adj_close - LME_AH_Close                         |       1 |
| US_Stock_XOM_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| LME_AH_Close - US_Stock_EWZ_adj_close                         |       1 |
| FX_EURAUD - LME_PB_Close                                      |       1 |
| LME_ZS_Close - FX_GBPUSD                                      |       1 |
| LME_CA_Close - FX_GBPJPY                                      |       1 |
| LME_AH_Close - US_Stock_ALB_adj_close                         |       1 |
| LME_PB_Close - FX_USDJPY                                      |       1 |
| LME_PB_Close - FX_NOKEUR                                      |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_EOG_adj_close  |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_TIP_adj_close      |       1 |
| US_Stock_RY_adj_close - LME_CA_Close                          |       1 |
| LME_ZS_Close - US_Stock_IAU_adj_close                         |       1 |
| LME_CA_Close - US_Stock_VWO_adj_close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - FX_CADJPY               |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_HL_adj_close       |       1 |
| FX_NOKJPY - LME_PB_Close                                      |       1 |
| LME_CA_Close - US_Stock_X_adj_close                           |       1 |
| LME_ZS_Close - FX_EURNZD                                      |       1 |
| LME_PB_Close - US_Stock_VTV_adj_close                         |       1 |
| US_Stock_DVN_adj_close - LME_PB_Close                         |       1 |
| FX_GBPNZD - JPX_Platinum_Standard_Futures_Close               |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_CVE_adj_close  |       1 |
| LME_ZS_Close - US_Stock_VALE_adj_close                        |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_FCX_adj_close  |       1 |
| US_Stock_JNK_adj_close - JPX_Platinum_Standard_Futures_Close  |       1 |
| LME_CA_Close - US_Stock_OKE_adj_close                         |       1 |
| US_Stock_RSP_adj_close - LME_CA_Close                         |       1 |
| LME_ZS_Close - US_Stock_CCJ_adj_close                         |       1 |
| LME_ZS_Close - US_Stock_FCX_adj_close                         |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_EWY_adj_close      |       1 |
| US_Stock_VWO_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| LME_PB_Close - FX_NOKUSD                                      |       1 |
| LME_CA_Close - US_Stock_VGK_adj_close                         |       1 |
| LME_PB_Close - LME_AH_Close                                   |       1 |
| US_Stock_IEMG_adj_close - JPX_Gold_Standard_Futures_Close     |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_VGIT_adj_close |       1 |
| FX_EURAUD - LME_CA_Close                                      |       1 |
| US_Stock_TECK_adj_close - LME_PB_Close                        |       1 |
| LME_AH_Close - US_Stock_CLF_adj_close                         |       1 |
| US_Stock_MPC_adj_close - LME_ZS_Close                         |       1 |
| LME_PB_Close - US_Stock_VEA_adj_close                         |       1 |
| JPX_Gold_Standard_Futures_Close - FX_EURCAD                   |       1 |
| JPX_Gold_Standard_Futures_Close - FX_NOKCHF                   |       1 |
| US_Stock_DE_adj_close - LME_AH_Close                          |       1 |
| FX_ZAREUR - LME_ZS_Close                                      |       1 |
| FX_ZARJPY - JPX_Gold_Standard_Futures_Close                   |       1 |
| US_Stock_XLE_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_BKR_adj_close      |       1 |
| US_Stock_LYB_adj_close - JPX_Platinum_Standard_Futures_Close  |       1 |
| LME_CA_Close - US_Stock_VYM_adj_close                         |       1 |
| LME_ZS_Close - US_Stock_KMI_adj_close                         |       1 |
| LME_CA_Close - FX_AUDCHF                                      |       1 |
| LME_CA_Close - US_Stock_CAT_adj_close                         |       1 |
| US_Stock_AMP_adj_close - LME_ZS_Close                         |       1 |
| LME_CA_Close - US_Stock_EWT_adj_close                         |       1 |
| LME_AH_Close - FX_EURJPY                                      |       1 |
| LME_CA_Close - US_Stock_HAL_adj_close                         |       1 |
| JPX_Gold_Standard_Futures_Close - FX_EURCHF                   |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_URA_adj_close  |       1 |
| US_Stock_EWY_adj_close - LME_ZS_Close                         |       1 |
| US_Stock_CVX_adj_close - LME_CA_Close                         |       1 |
| LME_PB_Close - FX_NZDUSD                                      |       1 |
| US_Stock_HES_adj_close - LME_PB_Close                         |       1 |
| FX_EURGBP - JPX_Platinum_Standard_Futures_Close               |       1 |
| LME_PB_Close - US_Stock_VGIT_adj_close                        |       1 |
| US_Stock_TD_adj_close - LME_ZS_Close                          |       1 |
| JPX_Gold_Standard_Futures_Close - FX_USDCHF                   |       1 |
| US_Stock_MS_adj_close - LME_AH_Close                          |       1 |
| LME_ZS_Close - US_Stock_SLV_adj_close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_WMB_adj_close  |       1 |
| FX_NOKEUR                                                     |       1 |
| LME_AH_Close - FX_NOKEUR                                      |       1 |
| LME_AH_Close - LME_PB_Close                                   |       1 |
| LME_PB_Close - LME_CA_Close                                   |       1 |
| LME_PB_Close - US_Stock_ALB_adj_close                         |       1 |
| LME_ZS_Close - US_Stock_EWZ_adj_close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_CLF_adj_close  |       1 |
| LME_CA_Close - FX_ZARJPY                                      |       1 |
| US_Stock_VT_adj_close - LME_ZS_Close                          |       1 |
| LME_AH_Close - US_Stock_SCCO_adj_close                        |       1 |
| US_Stock_FCX_adj_close - LME_ZS_Close                         |       1 |
| US_Stock_TRGP_adj_close - LME_CA_Close                        |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_MPC_adj_close      |       1 |
| US_Stock_RSP_adj_close - LME_PB_Close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_IAU_adj_close  |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_VXUS_adj_close |       1 |
| LME_PB_Close - US_Stock_LYB_adj_close                         |       1 |
| FX_CADJPY - LME_PB_Close                                      |       1 |
| US_Stock_ACWI_adj_close - JPX_Gold_Standard_Futures_Close     |       1 |
| LME_CA_Close - US_Stock_BNDX_adj_close                        |       1 |
| FX_ZAREUR - JPX_Gold_Standard_Futures_Close                   |       1 |
| LME_ZS_Close - US_Stock_IEMG_adj_close                        |       1 |
| FX_EURGBP - JPX_Gold_Standard_Futures_Close                   |       1 |
| US_Stock_OXY_adj_close - LME_CA_Close                         |       1 |
| US_Stock_RIO_adj_close - LME_AH_Close                         |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_XOM_adj_close      |       1 |
| LME_ZS_Close - US_Stock_HAL_adj_close                         |       1 |
| LME_CA_Close - US_Stock_VGIT_adj_close                        |       1 |
| LME_ZS_Close - US_Stock_JNK_adj_close                         |       1 |
| LME_AH_Close - US_Stock_VTV_adj_close                         |       1 |
| LME_PB_Close - US_Stock_TECK_adj_close                        |       1 |
| US_Stock_NUE_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| LME_ZS_Close - US_Stock_DE_adj_close                          |       1 |
| LME_PB_Close - US_Stock_VT_adj_close                          |       1 |
| US_Stock_VALE_adj_close - JPX_Gold_Standard_Futures_Close     |       1 |
| FX_ZARUSD - LME_AH_Close                                      |       1 |
| FX_USDJPY - JPX_Platinum_Standard_Futures_Close               |       1 |
| LME_AH_Close - FX_ZARCHF                                      |       1 |
| LME_PB_Close - FX_ZARGBP                                      |       1 |
| LME_AH_Close - US_Stock_CAT_adj_close                         |       1 |
| FX_EURCHF - LME_CA_Close                                      |       1 |
| US_Stock_OIH_adj_close - LME_CA_Close                         |       1 |
| LME_CA_Close - US_Stock_SLV_adj_close                         |       1 |
| LME_AH_Close - US_Stock_OKE_adj_close                         |       1 |
| US_Stock_KMI_adj_close - LME_CA_Close                         |       1 |
| US_Stock_VYM_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| LME_CA_Close - US_Stock_HES_adj_close                         |       1 |
| FX_NOKJPY - LME_CA_Close                                      |       1 |
| LME_ZS_Close - US_Stock_EFA_adj_close                         |       1 |
| JPX_Platinum_Standard_Futures_Close - FX_NOKCHF               |       1 |
| US_Stock_VT_adj_close                                         |       1 |
| JPX_Platinum_Standard_Futures_Close - US_Stock_ENB_adj_close  |       1 |
| LME_PB_Close - FX_AUDJPY                                      |       1 |
| US_Stock_COP_adj_close - JPX_Platinum_Standard_Futures_Close  |       1 |
| LME_ZS_Close - US_Stock_VEA_adj_close                         |       1 |
| JPX_Gold_Standard_Futures_Close - US_Stock_WMB_adj_close      |       1 |
| JPX_Platinum_Standard_Futures_Close - FX_CADCHF               |       1 |
| LME_PB_Close - FX_ZARCHF                                      |       1 |
| JPX_Platinum_Standard_Futures_Close - FX_NZDJPY               |       1 |
| FX_CHFJPY - LME_AH_Close                                      |       1 |
| US_Stock_VTV_adj_close - LME_ZS_Close                         |       1 |
| US_Stock_OIH_adj_close - JPX_Gold_Standard_Futures_Close      |       1 |
| LME_AH_Close - FX_ZARUSD                                      |       1 |
| JPX_Gold_Standard_Futures_Close - FX_USDJPY                   |       1 |
| LME_ZS_Close - FX_AUDCHF                                      |       1 |

ターゲットは、`pair`列で定義された金融商品（単一または価格差）を、`lag`列で指定された日数だけラグして定義されていると考えられます。

## 2. 価格差系列の生成と確認

### 2.1. 選択された代表的なターゲット (最初の5つ)

| target   |   lag | pair                                           |
|:---------|------:|:-----------------------------------------------|
| target_0 |     1 | US_Stock_VT_adj_close                          |
| target_1 |     1 | LME_PB_Close - US_Stock_VT_adj_close           |
| target_2 |     1 | LME_CA_Close - LME_ZS_Close                    |
| target_3 |     1 | LME_AH_Close - LME_ZS_Close                    |
| target_4 |     1 | LME_AH_Close - JPX_Gold_Standard_Futures_Close |

### 2.2. 価格差系列の生成

#### ターゲット: target_0 (`US_Stock_VT_adj_close` の `1` 日ラグ価格)
- 使用列: `US_Stock_VT_adj_close`
- 生成された系列の最初の5行:
|   date_id |   price_diff_target_0 |
|----------:|----------------------:|
|         0 |                   nan |
|         1 |                 63.4457 |
|         2 |                 63.7519 |
|         3 |                 64.3732 |
|         4 |                 64.6882 |

### 2.3. 生成した価格差系列の統計情報と欠損値

#### ターゲット: target_0
- 記述統計:
|