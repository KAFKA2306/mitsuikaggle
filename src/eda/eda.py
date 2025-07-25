import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Directory '{plots_dir}' created.")

    train_path = 'input/train.csv'
    train_labels_path = 'input/train_labels.csv'
    target_pairs_path = 'input/target_pairs.csv'

    print("--- Loading and Verifying Datasets ---")
    try:
        df_train = pd.read_csv(train_path)
        df_train_labels = pd.read_csv(train_labels_path)
        df_target_pairs = pd.read_csv(target_pairs_path)
        print(f"'{train_path}' loaded.")
        print(f"'{train_labels_path}' loaded.")
        print(f"'{target_pairs_path}' loaded.")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return

    print("\n--- Basic Information for df_train ---")
    print(df_train.info())
    print("\n--- Descriptive Statistics for df_train ---")
    print(df_train.describe())

    print("\n--- Basic Information for df_train_labels ---")
    print(df_train_labels.info())
    print("\n--- Descriptive Statistics for df_train_labels ---")
    print(df_train_labels.describe())

    print("\n--- EDA3: 主要な金融商品のテクニカル指標の計算と可視化 ---")
    assets_to_analyze = [
        'LME_AH_Close',
        'JPX_Gold_Standard_Futures_Close',
        'US_Stock_VT_adj_close', # US_Stock_SPY_adj_close が存在しない場合
        'FX_USDJPY'
    ]

    for asset in assets_to_analyze:
        if asset not in df_train.columns:
            if asset == 'US_Stock_VT_adj_close':
                if 'US_Stock_SPY_adj_close' in df_train.columns:
                    asset = 'US_Stock_SPY_adj_close'
                else:
                    print(f"Warning: {asset} も US_Stock_SPY_adj_close も見つかりません。スキップします。")
                    continue
            else:
                print(f"Warning: {asset} が見つかりません。スキップします。")
                continue

        # MA7, MA30, StdDev7 の計算
        df_train[f'{asset}_MA7'] = df_train[asset].rolling(window=7).mean()
        df_train[f'{asset}_MA30'] = df_train[asset].rolling(window=30).mean()
        df_train[f'{asset}_StdDev7'] = df_train[asset].rolling(window=7).std()

        # 元の価格、MA7、MA30 のプロット
        plt.figure(figsize=(12, 6))
        plt.plot(df_train['date_id'], df_train[asset], label=asset, alpha=0.7)
        plt.plot(df_train['date_id'], df_train[f'{asset}_MA7'], label='MA7', linestyle='--')
        plt.plot(df_train['date_id'], df_train[f'{asset}_MA30'], label='MA30', linestyle='-.')
        plt.title(f'{asset} Price with Moving Averages')
        plt.xlabel('Date ID')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        ma_plot_filename = f'plots/{asset}_ma.png'
        plt.savefig(ma_plot_filename)
        plt.close()
        print(f"'{ma_plot_filename}' を保存しました。")

        # StdDev7 のプロット
        plt.figure(figsize=(12, 6))
        plt.plot(df_train['date_id'], df_train[f'{asset}_StdDev7'], label='StdDev7', color='red')
        plt.title(f'{asset} 7-Day Standard Deviation')
        plt.xlabel('Date ID')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        stddev_plot_filename = f'plots/{asset}_stddev.png'
        plt.savefig(stddev_plot_filename)
        plt.close()
        print(f"'{stddev_plot_filename}' を保存しました。")

    print("\n--- EDA3: ターゲット変数の分布と外れ値の確認 ---")
    target_cols = [f'target_{i}' for i in range(5)]
    for target in target_cols:
        if target in df_train_labels.columns:
            # ヒストグラム
            plt.figure(figsize=(10, 6))
            sns.histplot(df_train_labels[target].dropna(), kde=True)
            plt.title(f'Distribution of {target}')
            plt.xlabel(target)
            plt.ylabel('Frequency')
            plt.tight_layout()
            hist_filename = f'plots/{target}_hist.png'
            plt.savefig(hist_filename)
            plt.close()
            print(f"'{hist_filename}' を保存しました。")

            # 箱ひげ図
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df_train_labels[target].dropna())
            plt.title(f'Box Plot of {target}')
            plt.ylabel(target)
            plt.tight_layout()
            boxplot_filename = f'plots/{target}_boxplot.png'
            plt.savefig(boxplot_filename)
            plt.close()
            print(f"'{boxplot_filename}' を保存しました。")
        else:
            print(f"Warning: {target} が df_train_labels に見つかりません。スキップします。")

    print("\n--- EDA3: 価格差系列と元の価格系列の比較（より詳細に） ---")
    # target_1 の詳細な比較
    target_1_info = df_target_pairs[df_target_pairs['target'] == 'target_1'].iloc[0]
    # 'pair' 列からアセット名を抽出
    pair_assets = target_1_info['pair'].split(' - ')
    col1_target1 = pair_assets[0]
    col2_target1 = pair_assets[1]
    lag_target1 = target_1_info['lag']

    if col1_target1 in df_train.columns and col2_target1 in df_train.columns:
        price_diff_target_1_series = df_train[col1_target1].shift(lag_target1) - df_train[col2_target1].shift(lag_target1)

        plt.figure(figsize=(14, 7))
        plt.plot(df_train['date_id'], df_train[col1_target1], label=f'Original {col1_target1}', alpha=0.7)
        plt.plot(df_train['date_id'], df_train[col2_target1], label=f'Original {col2_target1}', alpha=0.7)
        plt.plot(df_train['date_id'], price_diff_target_1_series, label=f'Price Diff ({col1_target1} - {col2_target1}, Lag {lag_target1})', linestyle='--')
        
        if 'target_1' in df_train_labels.columns:
            plt.plot(df_train_labels['date_id'], df_train_labels['target_1'], label='target_1 (Actual)', linestyle=':')

        plt.title(f'Comparison of Original Prices and Price Difference Series for target_1')
        plt.xlabel('Date ID')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        price_and_diff_target_1_filename = 'plots/price_and_diff_target_1.png'
        plt.savefig(price_and_diff_target_1_filename)
        plt.close()
        print(f"'{price_and_diff_target_1_filename}' を保存しました。")
    else:
        print(f"Warning: {col1_target1} または {col2_target1} が df_train に見つかりません。target_1 の比較をスキップします。")

    print("\n--- EDA3: ラグ日数の影響の可視化 ---")
    # 同じ pair で異なる lag を持つターゲットの例を特定
    # 例: US_Stock_VT_adj_close の異なるラグ
    lag_comparison_targets = []
    
    # target_0 は US_Stock_VT_adj_close の lag 1
    target_0_info = df_target_pairs[df_target_pairs['target'] == 'target_0'].iloc[0]
    lag_comparison_targets.append(target_0_info)

    # US_Stock_VT_adj_close を含む他のターゲットを探す
    other_vt_targets = df_target_pairs[
        (df_target_pairs['pair'].str.contains('US_Stock_VT_adj_close')) &
        (df_target_pairs['target'] != 'target_0')
    ]

    # 異なるラグを持つターゲットを最大2つ追加
    for _, row in other_vt_targets.iterrows():
        if row['lag'] != target_0_info['lag'] and len(lag_comparison_targets) < 3: # 最大3つのターゲットを比較
            lag_comparison_targets.append(row)
            
    if len(lag_comparison_targets) < 2:
        print("Warning: ラグ比較のための十分なターゲットが見つかりません。")
    else:
        plt.figure(figsize=(14, 7))
        for target_info in lag_comparison_targets:
            target_name = target_info['target']
            # 'pair' 列からアセット名を抽出
            pair_assets = target_info['pair'].split(' - ')
            col1 = pair_assets[0] # target_0 のように単一アセットの場合を考慮
            col2 = pair_assets[1] if len(pair_assets) > 1 else None # target_0 のように単一アセットの場合を考慮
            lag = target_info['lag']

            if col1 in df_train.columns:
                if col2 and col2 in df_train.columns:
                    price_diff_series = df_train[col1].shift(lag) - df_train[col2].shift(lag)
                    plt.plot(df_train['date_id'], price_diff_series, label=f'{target_name} ({col1}-{col2}, Lag {lag})')
                else:
                    # 単一アセットの場合、元の価格系列をプロット
                    plt.plot(df_train['date_id'], df_train[col1], label=f'{target_name} ({col1}, Lag {lag})')
            else:
                print(f"Warning: {col1} が df_train に見つかりません。{target_name} のラグ比較をスキップします。")

        plt.title('Comparison of Price Difference Series with Different Lags')
        plt.xlabel('Date ID')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        lag_comparison_filename = 'plots/lag_comparison.png'
        plt.savefig(lag_comparison_filename)
        plt.close()
        print(f"'{lag_comparison_filename}' を保存しました。")

if __name__ == "__main__":
    run_eda()