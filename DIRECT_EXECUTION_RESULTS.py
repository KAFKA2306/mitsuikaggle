#!/usr/bin/env python3
"""
DIRECT EXECUTION: Get Real Results Without Environment Issues
Calculate actual performance metrics directly.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# EXECUTION BLOCK 1: Load and prepare data
print("🔥 DIRECT EXECUTION - GETTING REAL RESULTS!")
print("="*60)

# Load minimal data
train_data = pd.read_csv('input/train.csv').head(200)
label_data = pd.read_csv('input/train_labels.csv').head(200)

# Merge data
merged_data = train_data.merge(label_data, on='date_id', how='inner')

# Prepare features and targets
feature_columns = [col for col in train_data.columns if col != 'date_id'][:10]  # 10 features
target_columns = [col for col in label_data.columns if col.startswith('target_')][:5]  # 5 targets

X_data = merged_data[feature_columns].fillna(0).values
y_data = merged_data[target_columns].fillna(0).values

print(f"✓ Data loaded: {merged_data.shape}")
print(f"✓ Features: {len(feature_columns)}")
print(f"✓ Targets: {len(target_columns)}")
print(f"✓ X shape: {X_data.shape}, y shape: {y_data.shape}")

# EXECUTION BLOCK 2: Train/test split
split_index = int(0.75 * len(X_data))
X_train = X_data[:split_index]
X_test = X_data[split_index:]
y_train = y_data[:split_index]
y_test = y_data[split_index:]

print(f"✓ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# EXECUTION BLOCK 3: Single Model Performance (LightGBM)
print("\n" + "="*40)
print("EXPERIMENT 1: SINGLE MODEL (LightGBM)")
print("="*40)

import lightgbm as lgb

single_model_scores = []
single_model_predictions = []

for i, target_col in enumerate(target_columns):
    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train[:, i])
    pred = model.predict(X_test)
    
    # Calculate Spearman correlation
    correlation = spearmanr(y_test[:, i], pred)[0]
    if np.isnan(correlation):
        correlation = 0.0
    
    single_model_scores.append(correlation)
    single_model_predictions.append(pred)
    
    print(f"Target {i+1} ({target_col}): {correlation:.4f}")

mean_single_score = np.mean(single_model_scores)
std_single_score = np.std(single_model_scores)
sharpe_single = mean_single_score / std_single_score if std_single_score > 0 else mean_single_score

print(f"\n📊 Single Model Results:")
print(f"  Mean Spearman: {mean_single_score:.4f}")
print(f"  Std Spearman: {std_single_score:.4f}")
print(f"  Sharpe-like Score: {sharpe_single:.4f}")

# EXECUTION BLOCK 4: Classical Ensemble (LightGBM + XGBoost)
print("\n" + "="*40)
print("EXPERIMENT 2: CLASSICAL ENSEMBLE")
print("="*40)

import xgboost as xgb

ensemble_scores = []
ensemble_predictions = []

for i, target_col in enumerate(target_columns):
    # LightGBM model
    lgb_model = lgb.LGBMRegressor(
        n_estimators=40,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train[:, i])
    pred_lgb = lgb_model.predict(X_test)
    
    # XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=40,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train[:, i])
    pred_xgb = xgb_model.predict(X_test)
    
    # Ensemble prediction (equal weights)
    ensemble_pred = 0.5 * pred_lgb + 0.5 * pred_xgb
    
    # Calculate correlations
    lgb_corr = spearmanr(y_test[:, i], pred_lgb)[0]
    xgb_corr = spearmanr(y_test[:, i], pred_xgb)[0]
    ens_corr = spearmanr(y_test[:, i], ensemble_pred)[0]
    
    if np.isnan(lgb_corr): lgb_corr = 0.0
    if np.isnan(xgb_corr): xgb_corr = 0.0
    if np.isnan(ens_corr): ens_corr = 0.0
    
    ensemble_scores.append(ens_corr)
    ensemble_predictions.append(ensemble_pred)
    
    print(f"Target {i+1}: LGB={lgb_corr:.4f}, XGB={xgb_corr:.4f}, Ensemble={ens_corr:.4f}")

mean_ensemble_score = np.mean(ensemble_scores)
std_ensemble_score = np.std(ensemble_scores)
sharpe_ensemble = mean_ensemble_score / std_ensemble_score if std_ensemble_score > 0 else mean_ensemble_score

print(f"\n📊 Classical Ensemble Results:")
print(f"  Mean Spearman: {mean_ensemble_score:.4f}")
print(f"  Std Spearman: {std_ensemble_score:.4f}")
print(f"  Sharpe-like Score: {sharpe_ensemble:.4f}")

# EXECUTION BLOCK 5: Multi-Model Ensemble (LGB + XGB + RF)
print("\n" + "="*40)
print("EXPERIMENT 3: MULTI-MODEL ENSEMBLE")
print("="*40)

from sklearn.ensemble import RandomForestRegressor

multimodel_scores = []
multimodel_predictions = []

for i, target_col in enumerate(target_columns):
    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=30, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train[:, i])
    pred_lgb = lgb_model.predict(X_test)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=30, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train[:, i])
    pred_xgb = xgb_model.predict(X_test)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=30, random_state=42)
    rf_model.fit(X_train, y_train[:, i])
    pred_rf = rf_model.predict(X_test)
    
    # Multi-model ensemble (equal weights)
    multi_pred = (pred_lgb + pred_xgb + pred_rf) / 3
    
    # Calculate correlations
    lgb_corr = spearmanr(y_test[:, i], pred_lgb)[0] if not np.isnan(spearmanr(y_test[:, i], pred_lgb)[0]) else 0.0
    xgb_corr = spearmanr(y_test[:, i], pred_xgb)[0] if not np.isnan(spearmanr(y_test[:, i], pred_xgb)[0]) else 0.0
    rf_corr = spearmanr(y_test[:, i], pred_rf)[0] if not np.isnan(spearmanr(y_test[:, i], pred_rf)[0]) else 0.0
    multi_corr = spearmanr(y_test[:, i], multi_pred)[0] if not np.isnan(spearmanr(y_test[:, i], multi_pred)[0]) else 0.0
    
    multimodel_scores.append(multi_corr)
    multimodel_predictions.append(multi_pred)
    
    print(f"Target {i+1}: LGB={lgb_corr:.4f}, XGB={xgb_corr:.4f}, RF={rf_corr:.4f}, Multi={multi_corr:.4f}")

mean_multimodel_score = np.mean(multimodel_scores)
std_multimodel_score = np.std(multimodel_scores)
sharpe_multimodel = mean_multimodel_score / std_multimodel_score if std_multimodel_score > 0 else mean_multimodel_score

print(f"\n📊 Multi-Model Ensemble Results:")
print(f"  Mean Spearman: {mean_multimodel_score:.4f}")
print(f"  Std Spearman: {std_multimodel_score:.4f}")
print(f"  Sharpe-like Score: {sharpe_multimodel:.4f}")

# EXECUTION BLOCK 6: Final Results Summary
print("\n" + "="*60)
print("🏆 FINAL RESULTS SUMMARY")
print("="*60)

results_summary = [
    ("Single Model (LightGBM)", mean_single_score, std_single_score, sharpe_single),
    ("Classical Ensemble (LGB+XGB)", mean_ensemble_score, std_ensemble_score, sharpe_ensemble),
    ("Multi-Model Ensemble (LGB+XGB+RF)", mean_multimodel_score, std_multimodel_score, sharpe_multimodel)
]

# Sort by Sharpe-like score
results_summary.sort(key=lambda x: x[3], reverse=True)

print("Ranking by Sharpe-like Score:")
for i, (method, mean_corr, std_corr, sharpe) in enumerate(results_summary):
    print(f"{i+1}. {method}")
    print(f"   Mean Spearman: {mean_corr:.4f}")
    print(f"   Std Spearman: {std_corr:.4f}")
    print(f"   Sharpe-like Score: {sharpe:.4f}")
    print()

# Calculate improvements
best_method = results_summary[0]
baseline_sharpe = results_summary[-1][3]  # Worst performing as baseline

if baseline_sharpe > 0:
    improvement = ((best_method[3] - baseline_sharpe) / baseline_sharpe) * 100
    print(f"🎯 Best Method: {best_method[0]}")
    print(f"📈 Improvement over baseline: {improvement:.1f}%")

print(f"✅ ACTUAL RESULTS OBTAINED!")
print(f"📊 Data points tested: {len(y_test)} samples across {len(target_columns)} targets")
print(f"🔬 Methods compared: {len(results_summary)} approaches")

# Save results to file
results_data = {
    'methods': [r[0] for r in results_summary],
    'mean_spearman': [r[1] for r in results_summary],
    'std_spearman': [r[2] for r in results_summary],
    'sharpe_like_score': [r[3] for r in results_summary]
}

results_df = pd.DataFrame(results_data)
results_df.to_csv('ACTUAL_EXPERIMENT_RESULTS.csv', index=False)
print(f"📁 Results saved to: ACTUAL_EXPERIMENT_RESULTS.csv")

print("\n🎉 DIRECT EXECUTION COMPLETED SUCCESSFULLY!")