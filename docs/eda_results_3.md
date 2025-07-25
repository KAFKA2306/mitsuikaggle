# EDA3 Results

## 1. Technical Indicators of Major Financial Products

Time series plots showing the original price, 7-day Moving Average (MA7), and 30-day Moving Average (MA30) for selected financial products. Also, 7-day Standard Deviation (StdDev7) plots are included.

### LME_AH_Close Price Trends and Moving Averages

![LME_AH_Close MA](plots/LME_AH_Close_ma.png)

### LME_AH_Close 7-Day Standard Deviation

![LME_AH_Close StdDev](plots/LME_AH_Close_stddev.png)

### JPX_Gold_Standard_Futures_Close Price Trends and Moving Averages

![JPX_Gold_Standard_Futures_Close MA](plots/JPX_Gold_Standard_Futures_Close_ma.png)

### JPX_Gold_Standard_Futures_Close 7-Day Standard Deviation

![JPX_Gold_Standard_Futures_Close StdDev](plots/JPX_Gold_Standard_Futures_Close_stddev.png)

### US_Stock_VT_adj_close Price Trends and Moving Averages

![US_Stock_VT_adj_close MA](plots/US_Stock_VT_adj_close_ma.png)

### US_Stock_VT_adj_close 7-Day Standard Deviation

![US_Stock_VT_adj_close StdDev](plots/US_Stock_VT_adj_close_stddev.png)

### FX_USDJPY Price Trends and Moving Averages

![FX_USDJPY MA](plots/FX_USDJPY_ma.png)

### FX_USDJPY 7-Day Standard Deviation

![FX_USDJPY StdDev](plots/FX_USDJPY_stddev.png)

## 2. Distribution and Outliers of Target Variables

Histograms and box plots for the first five target variables (`target_0` to `target_4`).

### Target 0 Distribution

![Target 0 Histogram](plots/target_0_hist.png)

### Target 0 Outliers

![Target 0 Boxplot](plots/target_0_boxplot.png)

### Target 1 Distribution

![Target 1 Histogram](plots/target_1_hist.png)

### Target 1 Outliers

![Target 1 Boxplot](plots/target_1_boxplot.png)

### Target 2 Distribution

![Target 2 Histogram](plots/target_2_hist.png)

### Target 2 Outliers

![Target 2 Boxplot](plots/target_2_boxplot.png)

### Target 3 Distribution

![Target 3 Histogram](plots/target_3_hist.png)

### Target 3 Outliers

![Target 3 Boxplot](plots/target_3_boxplot.png)

### Target 4 Distribution

![Target 4 Histogram](plots/target_4_hist.png)

### Target 4 Outliers

![Target 4 Boxplot](plots/target_4_boxplot.png)

## 3. Comparison of Price Difference Series and Original Price Series

This plot compares the original price series of `LME_PB_Close` and `US_Stock_VT_adj_close` with their generated price difference series (`target_1`).

### Original Prices vs. Price Difference for Target 1

![Price and Diff Target 1](plots/price_and_diff_target_1.png)

## 4. Visualization of Lag Day Impact

This plot compares the behavior of price difference series with different lag days for the same asset pair.

### Lag Comparison for US_Stock_VT_adj_close (Target 0 vs. other lags if available)

![Lag Comparison](plots/lag_comparison.png)