#!/usr/bin/env python3
"""
EXECUTION: Minimal Working Experiment
Get actual results with smallest possible dataset.
"""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 MINIMAL EXECUTION STARTING...")
    print("="*50)
    
    # Step 1: Load minimal data
    print("Step 1: Loading minimal data...")
    train = pd.read_csv('input/train.csv').head(100)  # Only 100 rows
    labels = pd.read_csv('input/train_labels.csv').head(100)
    print(f"✓ Loaded: train={train.shape}, labels={labels.shape}")
    
    # Step 2: Merge and prepare
    print("Step 2: Data preparation...")
    merged = train.merge(labels, on='date_id', how='inner')
    print(f"✓ Merged: {merged.shape}")
    
    # Get 5 features and 3 targets
    feature_cols = [col for col in train.columns if col != 'date_id'][:5]
    target_cols = [col for col in labels.columns if col.startswith('target_')][:3]
    
    X = merged[feature_cols].fillna(0).values
    y = merged[target_cols].fillna(0).values
    print(f"✓ Arrays: X={X.shape}, y={y.shape}")
    
    # Step 3: Train/test split
    print("Step 3: Train/test split...")
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 4: Simple model test
    print("Step 4: Testing LightGBM...")
    try:
        import lightgbm as lgb
        from scipy.stats import spearmanr
        
        scores = []
        for i in range(3):  # 3 targets
            # Train model
            model = lgb.LGBMRegressor(n_estimators=20, verbose=-1, random_state=42)
            model.fit(X_train, y_train[:, i])
            
            # Predict
            pred = model.predict(X_test)
            
            # Evaluate
            corr = spearmanr(y_test[:, i], pred)[0]
            if np.isnan(corr):
                corr = 0.0
            scores.append(corr)
            
            print(f"  Target {i+1}: {corr:.4f}")
        
        mean_score = np.mean(scores)
        print(f"✓ LightGBM Mean Spearman: {mean_score:.4f}")
        
    except Exception as e:
        print(f"✗ LightGBM failed: {e}")
        return False
    
    # Step 5: Test ensemble
    print("Step 5: Testing simple ensemble...")
    try:
        import xgboost as xgb
        
        ensemble_scores = []
        for i in range(3):
            # LightGBM
            lgb_model = lgb.LGBMRegressor(n_estimators=15, verbose=-1, random_state=42)
            lgb_model.fit(X_train, y_train[:, i])
            pred_lgb = lgb_model.predict(X_test)
            
            # XGBoost  
            xgb_model = xgb.XGBRegressor(n_estimators=15, verbosity=0, random_state=42)
            xgb_model.fit(X_train, y_train[:, i])
            pred_xgb = xgb_model.predict(X_test)
            
            # Ensemble (equal weights)
            ensemble_pred = 0.5 * pred_lgb + 0.5 * pred_xgb
            
            # Evaluate
            lgb_corr = spearmanr(y_test[:, i], pred_lgb)[0]
            xgb_corr = spearmanr(y_test[:, i], pred_xgb)[0]
            ens_corr = spearmanr(y_test[:, i], ensemble_pred)[0]
            
            if np.isnan(lgb_corr): lgb_corr = 0.0
            if np.isnan(xgb_corr): xgb_corr = 0.0
            if np.isnan(ens_corr): ens_corr = 0.0
            
            ensemble_scores.append(ens_corr)
            print(f"  Target {i+1}: LGB={lgb_corr:.4f}, XGB={xgb_corr:.4f}, Ensemble={ens_corr:.4f}")
        
        mean_ensemble = np.mean(ensemble_scores)
        print(f"✓ Ensemble Mean Spearman: {mean_ensemble:.4f}")
        
        # Calculate competition metric (simplified)
        print("Step 6: Competition metric...")
        if len(ensemble_scores) > 1:
            std_score = np.std(ensemble_scores)
            sharpe_like = mean_ensemble / std_score if std_score > 0 else mean_ensemble
            print(f"✓ Sharpe-like score: {sharpe_like:.4f}")
        
    except Exception as e:
        print(f"✗ Ensemble failed: {e}")
    
    print("="*50)
    print("🎉 MINIMAL EXECUTION COMPLETED!")
    print(f"📊 Results Summary:")
    print(f"  - Single Model (LGB): {mean_score:.4f}")
    if 'mean_ensemble' in locals():
        print(f"  - Ensemble (LGB+XGB): {mean_ensemble:.4f}")
        improvement = ((mean_ensemble - mean_score) / mean_score * 100) if mean_score > 0 else 0
        print(f"  - Improvement: {improvement:.1f}%")
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"Exit code: {0 if success else 1}")