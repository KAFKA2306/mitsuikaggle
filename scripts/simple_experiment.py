#!/usr/bin/env python3
"""
Simple experiment validation
"""

import pandas as pd
import numpy as np
import json
import traceback

def run_simple_experiment():
    try:
        # Write to file for output
        with open('results/experiments/experiment_log.txt', 'w') as log:
            log.write("🚀 SIMPLE EXPERIMENT STARTED\n")
            log.write("="*40 + "\n")
            
            # Load data
            log.write("Loading data...\n")
            train_data = pd.read_csv('input/train.csv')
            train_labels = pd.read_csv('input/train_labels.csv')
            
            log.write(f"Training data: {train_data.shape}\n")
            log.write(f"Training labels: {train_labels.shape}\n")
            
            # Basic preprocessing
            if 'date_id' in train_data.columns:
                X = train_data.drop('date_id', axis=1)
            else:
                X = train_data
                
            if 'date_id' in train_labels.columns:
                y = train_labels.drop('date_id', axis=1)
            else:
                y = train_labels
            
            # Fill missing values
            X = X.fillna(0)
            y = y.fillna(0)
            
            # Select numeric columns only
            X = X.select_dtypes(include=[np.number])
            y = y.select_dtypes(include=[np.number])
            
            # Align data
            min_samples = min(len(X), len(y))
            X = X.iloc[:min_samples]
            y = y.iloc[:min_samples]
            
            log.write(f"Processed data: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets\n")
            
            # Simple baseline test: Calculate correlation between first feature and first target
            if len(X) > 0 and len(y) > 0:
                first_feature = X.iloc[:, 0].values
                first_target = y.iloc[:, 0].values
                
                # Remove any infinite or NaN values
                mask = np.isfinite(first_feature) & np.isfinite(first_target)
                first_feature = first_feature[mask]
                first_target = first_target[mask]
                
                if len(first_feature) > 1:
                    correlation = np.corrcoef(first_feature, first_target)[0, 1]
                    if not np.isnan(correlation):
                        log.write(f"Sample correlation: {correlation:.4f}\n")
                    else:
                        log.write("Sample correlation: NaN\n")
                else:
                    log.write("Insufficient data for correlation\n")
            
            # Test feature engineering
            log.write("\nTesting feature engineering...\n")
            
            # Add momentum features
            X_enhanced = X.copy()
            for i, col in enumerate(X.columns[:5]):  # First 5 columns
                pct_change = X[col].pct_change().fillna(0)
                X_enhanced[f'{col}_momentum'] = pct_change
                
                rolling_mean = X[col].rolling(5).mean().fillna(X[col])
                X_enhanced[f'{col}_ma5'] = rolling_mean
            
            log.write(f"Enhanced features: {X.shape[1]} → {X_enhanced.shape[1]} (+{X_enhanced.shape[1] - X.shape[1]})\n")
            
            # Calculate some basic statistics
            original_std = X.std().mean()
            enhanced_std = X_enhanced.std().mean()
            
            log.write(f"Feature variability - Original: {original_std:.4f}, Enhanced: {enhanced_std:.4f}\n")
            
            # Simple ensemble test
            log.write("\nTesting simple ensemble...\n")
            
            # Create 3 different predictions using different features
            predictions = []
            for i in range(3):
                if i < X.shape[1]:
                    pred = X.iloc[:, i] * 0.1  # Simple linear transform
                    predictions.append(pred.values)
            
            if len(predictions) >= 3:
                # Average ensemble
                ensemble_pred = np.mean(predictions, axis=0)
                log.write(f"Ensemble prediction range: {ensemble_pred.min():.4f} to {ensemble_pred.max():.4f}\n")
            
            # Results summary
            results = {
                'experiment': 'simple_validation',
                'data_shape': {'samples': int(X.shape[0]), 'features': int(X.shape[1]), 'targets': int(y.shape[1])},
                'enhanced_features': int(X_enhanced.shape[1]),
                'feature_improvement': int(X_enhanced.shape[1] - X.shape[1]),
                'original_variability': float(original_std),
                'enhanced_variability': float(enhanced_std),
                'status': 'completed'
            }
            
            # Save results
            with open('results/experiments/simple_experiment_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            log.write("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY\n")
            log.write(f"Results saved to simple_experiment_results.json\n")
            
        return results
        
    except Exception as e:
        error_msg = f"Error in experiment: {str(e)}\n{traceback.format_exc()}"
        with open('results/experiments/experiment_error.txt', 'w') as f:
            f.write(error_msg)
        return {'error': str(e)}

if __name__ == "__main__":
    result = run_simple_experiment()
    print("Experiment completed - check log files")