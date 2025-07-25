import pandas as pd
import numpy as np
import io

def run_eda_new():
    output = io.StringIO()

    output.write("# EDA2 結果\n\n")

    # 1. target_pairs.csvの分析
    target_pairs_path = 'input/target_pairs.csv'
    try:
        df_target_pairs = pd.read_csv(target_pairs_path)
        output.write(f"## 1. `target_pairs.csv`の分析\n\n")
        output.write(f"### 1.1. `{target_pairs_path}`の基本的な情報\n\n")
        output.write("#### df.info()\n")
        df_target_pairs.info(buf=output)
        output.write("\n\n")

        output.write("#### df.head()\n")
        output.write(df_target_pairs.head().to_markdown(index=False))
        output.write("\n\n")

        output.write("#### df.describe()\n")
        output.write(df_target_pairs.describe().to_markdown())
        output.write("\n\n")

### 1.2. `lag`と`pair`列のユニーク値と分布\n\n")
        for col in ['lag', 'pair']:
            output.write(f"#### 列 `{col}`\n")
            output.write(f"- ユニーク値の数: {df_target_pairs[col].nunique()}\n")
            output.write(f"- 分布:\n")
            output.write(df_target_pairs[col].value_counts().to_markdown())
            output.write("\n\n")
        
        output.write("ターゲットは、`pair`列で定義された金融商品（単一または価格差）を、`lag`列で指定された日数だけラグして定義されていると考えられます。\n\n")

    except FileNotFoundError as e:
        output.write(f"エラー: ファイルが見つかりません - {e}\n")
        return output.getvalue()

    # 2. 価格差系列の生成と確認
    output.write("## 2. 価格差系列の生成と確認\n\n")

    train_path = 'input/train.csv'
    train_labels_path = 'input/train_labels.csv'

    try:
        df_train = pd.read_csv(train_path)
        df_train_labels = pd.read_csv(train_labels_path)
    except FileNotFoundError as e:
        output.write(f"エラー: ファイルが見つかりません - {e}\n")
        return output.getvalue()

    # 代表的なターゲットの選択 (最初の5つ)
    selected_targets = df_target_pairs.head(5)
    output.write("### 2.1. 選択された代表的なターゲット (最初の5つ)\n\n")
    output.write(selected_targets.to_markdown(index=False))
    output.write("\n\n")

    # 価格列の特定
    # train.csvの列名から価格列を推測
    # 'LME_AH_Close', 'LME_CA_Close', 'LME_PB_Close', 'LME_ZS_Close'
    # 'JPX_Gold_Mini_Futures_Close', 'JPX_Gold_Rolling-Spot_Futures_Close', 'JPX_Gold_Standard_Futures_Close',
    # 'JPX_Platinum_Mini_Futures_Close', 'JPX_Platinum_Standard_Futures_Close', 'JPX_RSS3_Rubber_Futures_Close'
    # 'US_Stock_XXXX_adj_close'
    # 'FX_XXXX'
    
    # 簡略化のため、ここではCloseを含む列とFX列を対象とする
    price_cols = [col for col in df_train.columns if 'Close' in col or 'FX_' in col]
    
    if not price_cols:
        output.write("エラー: `train.csv`に適切な価格列が見つかりませんでした。`LME_AH_Close`や`FX_ZARGBP`のような列名を想定していましたが、見つかりませんでした。\n")
        return output.getvalue()

    output.write("### 2.2. 価格差系列の生成\n\n")
    
    price_diff_series = {}
    correlations = {}

    for index, row in selected_targets.iterrows():
        target_name = row['target']
        date_delta = row['lag']
        pair_str = row['pair']

        asset_a = None
        asset_b = None

        if ' - ' in pair_str:
            parts = pair_str.split(' - ')
            asset_a = parts[0].strip()
            asset_b = parts[1].strip()
        else:
            # 単一の資産の場合、asset_bのみを使用し、asset_aはNoneとする
            asset_b = pair_str.strip()

        col_a = None
        col_b = None

        # asset_aに対応する列名を探す
        if asset_a:
            if asset_a in df_train.columns:
                col_a = asset_a
            else:
                for p_col in price_cols:
                    if asset_a in p_col:
                        col_a = p_col
                        break
            if col_a is None:
                output.write(f"警告: asset_a '{asset_a}' に対応する価格列が`train.csv`に見つかりませんでした。このターゲットはスキップします。\n")
                continue

        # asset_bに対応する列名を探す
        if asset_b:
            if asset_b in df_train.columns:
                col_b = asset_b
            else:
                for p_col in price_cols:
                    if asset_b in p_col:
                        col_b = p_col
                        break
            if col_b is None:
                output.write(f"警告: asset_b '{asset_b}' に対応する価格列が`train.csv`に見つかりませんでした。このターゲットはスキップします。\n")
                continue
        
        # 価格差系列の計算
        df_temp = df_train[['date_id']].copy()
        
        if col_a and col_b:
            # 価格差 (asset_b - asset_a) のラグ値
            df_temp[f'{col_a}_lag'] = df_train.groupby('date_id')[col_a].shift(date_delta)
            df_temp[f'{col_b}_lag'] = df_train.groupby('date_id')[col_b].shift(date_delta)
            df_temp[f'price_diff_{target_name}'] = df_temp[f'{col_b}_lag'] - df_temp[f'{col_a}_lag']
            description_str = f"(`{asset_a}` と `{asset_b}` の `{date_delta}` 日ラグ価格差)"
            # 価格差系列の計算
            # df_trainはdate_idでソートされていると仮定し、DataFrame全体にshiftを適用
            
            # 必要な列を抽出
            cols_to_extract = ['date_id']
            if col_a:
                cols_to_extract.append(col_a)
            if col_b:
                cols_to_extract.append(col_b)
            
            df_temp = df_train[cols_to_extract].copy()
    
            if col_a and col_b:
                # 価格差 (asset_b - asset_a) のラグ値
                df_temp[f'{col_a}_lag'] = df_temp[col_a].shift(date_delta)
                df_temp[f'{col_b}_lag'] = df_temp[col_b].shift(date_delta)
                df_temp[f'price_diff_{target_name}'] = df_temp[f'{col_b}_lag'] - df_temp[f'{col_a}_lag']
                description_str = f"(`{asset_a}` と `{asset_b}` の `{date_delta}` 日ラグ価格差)"
                used_cols_str = f"`{col_a}`, `{col_b}`"
            elif col_b:
                # 単一資産のラグ値
                df_temp[f'{col_b}_lag'] = df_temp[col_b].shift(date_delta)
                df_temp[f'price_diff_{target_name}'] = df_temp[f'{col_b}_lag']
                description_str = f"(`{asset_b}` の `{date_delta}` 日ラグ価格)"
                used_cols_str = f"`{col_b}`"
            else:
                output.write(f"警告: ターゲット '{target_name}' の価格系列を生成できませんでした。このターゲットはスキップします。\n")
                continue
    
            price_diff_series[target_name] = df_temp[['date_id', f'price_diff_{target_name}']]
            
            output.write(f"#### ターゲット: {target_name} {description_str}\n")
            output.write(f"- 使用列: {used_cols_str}\n")
            output.write(f"- 生成された系列の最初の5行:\n")
            output.write(price_diff_series[target_name].head().to_markdown(index=False))
            output.write("\n\n")
    
            output.write(f"### 2.3. 生成した価格差系列の統計情報と欠損値\n\n")
            output.write(f"#### ターゲット: {target_name}\n")
            output.write(f"- 記述統計:\n")
            output.write(price_diff_series[target_name][f'price_diff_{target_name}'].describe().to_markdown())
            output.write("\n\n")
            output.write(f"- 欠損値の数: {price_diff_series[target_name][f'price_diff_{target_name}'].isnull().sum()}\n")
            output.write("\n")
    
            # 相関の計算
            # train_labelsとdate_idで結合し、対応するターゲット列との相関を計算
            merged_df = pd.merge(df_train_labels[['date_id', target_name]], price_diff_series[target_name], on='date_id', how='inner')
            
            if not merged_df.empty:
                correlation = merged_df[target_name].corr(merged_df[f'price_diff_{target_name}'])
                correlations[target_name] = correlation
                output.write(f"### 2.4. 生成した価格差系列とターゲット列の相関\n\n")
                output.write(f"#### ターゲット: {target_name}\n")
                output.write(f"- 相関 (`{target_name}` と `price_diff_{target_name}`): {correlation:.4f}\n")
                output.write("\n")
            else:
                output.write(f"警告: ターゲット '{target_name}' の価格差系列とラベルデータとの結合に失敗しました。相関は計算できませんでした。\n")
                output.write("\n")

    return output.getvalue()

if __name__ == "__main__":
    eda_results = run_eda_new()
    print(eda_results)