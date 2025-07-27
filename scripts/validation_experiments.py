#!/usr/bin/env python3
"""
Validation Experiments for Ultra Prediction Accuracy
Simplified version to verify improvement strategies
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import json
from sklearn.preprocessing import StandardScaler

def main():
    print("🚀 ULTRA PREDICTION VALIDATION EXPERIMENTS")
    print("="*50)
    
    # Load data
    print("📊 Loading data...")
    train_data = pd.read_csv('input/train.csv')
    train_labels = pd.read_csv('input/train_labels.csv')
    
    print(f"   Training data: {train_data.shape}")
    print(f"   Training labels: {train_labels.shape}")
    
    # Quick preprocessing
    if 'date_id' in train_data.columns:
        X = train_data.drop('date_id', axis=1).fillna(0)
    else:
        X = train_data.fillna(0)
        
    if 'date_id' in train_labels.columns:
        y = train_labels.drop('date_id', axis=1).fillna(0)
    else:
        y = train_labels.fillna(0)
    
    # Align data
    min_samples = min(len(X), len(y))
    X = X.iloc[:min_samples].select_dtypes(include=[np.number])
    y = y.iloc[:min_samples].select_dtypes(include=[np.number])
    
    print(f"   Aligned: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split data
    split_idx = int(0.8 * len(X_scaled))
    X_train = X_scaled[:split_idx]
    y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]
    y_val = y_scaled[split_idx:]
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Simple baseline calculation
    print("\n🧪 EXPERIMENT 1: BASELINE CORRELATION")
    print("-" * 30)
    
    correlations = []
    for i in range(min(50, y_val.shape[1])):  # Test first 50 targets
        # Simple prediction: mean of training
        y_pred_simple = np.full(len(y_val), np.mean(y_train[:, i]))
        
        corr = np.corrcoef(y_val[:, i], y_pred_simple)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    if len(correlations) > 0:
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        baseline_sharpe = mean_corr / std_corr if std_corr > 0 else 0
        print(f"   Simple Baseline Sharpe: {baseline_sharpe:.4f}")
    else:
        baseline_sharpe = 0
        print("   No valid correlations computed")
    
    # Feature engineering test
    print("\n🧪 EXPERIMENT 2: FEATURE ENGINEERING")
    print("-" * 35)
    
    # Add simple engineered features
    X_enhanced = X.copy()
    
    # Add momentum features (% change)
    for col in X.columns[:10]:  # First 10 columns
        X_enhanced[f'{col}_pct_change'] = X[col].pct_change().fillna(0)
        X_enhanced[f'{col}_rolling_mean_5'] = X[col].rolling(5).mean().fillna(X[col])
    
    # Add volatility features
    for col in X.columns[:5]:  # First 5 columns
        X_enhanced[f'{col}_volatility'] = X[col].rolling(10).std().fillna(0)
    
    print(f"   Enhanced features: {X.shape[1]} → {X_enhanced.shape[1]} (+{X_enhanced.shape[1] - X.shape[1]})")
    
    # Test with enhanced features
    X_enh_scaled = scaler_X.fit_transform(X_enhanced.fillna(0))
    X_enh_val = X_enh_scaled[split_idx:]
    
    enhanced_correlations = []
    for i in range(min(50, y_val.shape[1])):
        # Use first enhanced feature as simple predictor
        if X_enh_val.shape[1] > X.shape[1]:
            y_pred_enh = X_enh_val[:, X.shape[1]]  # First engineered feature
            corr = np.corrcoef(y_val[:, i], y_pred_enh)[0, 1]
            if not np.isnan(corr):
                enhanced_correlations.append(corr)
    
    if len(enhanced_correlations) > 0:
        enh_mean_corr = np.mean(enhanced_correlations)
        enh_std_corr = np.std(enhanced_correlations)
        enhanced_sharpe = enh_mean_corr / enh_std_corr if enh_std_corr > 0 else 0
        improvement = ((enhanced_sharpe - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe != 0 else 0
        print(f"   Enhanced Sharpe: {enhanced_sharpe:.4f} ({improvement:+.1f}%)")
    else:
        enhanced_sharpe = 0
        print("   No enhanced correlations computed")
    
    # Summary
    print("\n🏆 VALIDATION SUMMARY")
    print("="*30)
    print(f"Baseline Sharpe:  {baseline_sharpe:.4f}")
    print(f"Enhanced Sharpe:  {enhanced_sharpe:.4f}")
    
    if baseline_sharpe > 0:
        improvement = ((enhanced_sharpe - baseline_sharpe) / baseline_sharpe) * 100
        print(f"Improvement:      {improvement:+.1f}%")
    
    # Save results
    results = {
        'baseline_sharpe': float(baseline_sharpe),
        'enhanced_sharpe': float(enhanced_sharpe),
        'original_features': int(X.shape[1]),
        'enhanced_features': int(X_enhanced.shape[1]),
        'samples': int(X.shape[0]),
        'targets': int(y.shape[1])
    }
    
    with open('results/experiments/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Validation experiments completed!")
    return results

if __name__ == "__main__":
    results = main()