#!/usr/bin/env python3
"""
Minimal experiment to test basic functionality without complex framework.
"""

import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run minimal experiment."""
    logger.info("Starting minimal experiment...")
    
    try:
        # Load data with minimal processing
        logger.info("Loading data...")
        train_data = pd.read_csv('input/train.csv').head(200)  # Small sample
        train_labels = pd.read_csv('input/train_labels.csv').head(200)
        
        logger.info(f"Data loaded: train={train_data.shape}, labels={train_labels.shape}")
        
        # Simple merge
        merged = train_data.merge(train_labels, on='date_id', how='inner')
        logger.info(f"Merged data: {merged.shape}")
        
        # Get a few features and targets
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:5]
        target_cols = [col for col in train_labels.columns if col.startswith('target_')][:3]
        
        logger.info(f"Using {len(feature_cols)} features and {len(target_cols)} targets")
        
        # Prepare arrays
        X = merged[feature_cols].fillna(0).values
        y = merged[target_cols].fillna(0).values
        
        logger.info(f"Arrays prepared: X={X.shape}, y={y.shape}")
        
        # Simple LightGBM test
        import lightgbm as lgb
        
        # Train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info("Training models...")
        scores = []
        
        for i, target_name in enumerate(target_cols):
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train[:, i])
            model = lgb.train(
                {'objective': 'regression', 'verbose': -1},
                train_data,
                num_boost_round=50
            )
            
            # Predict
            pred = model.predict(X_test)
            score = spearmanr(y_test[:, i], pred)[0]
            if np.isnan(score):
                score = 0.0
            scores.append(score)
            
            logger.info(f"Target {i+1} ({target_name}): {score:.4f}")
        
        # Overall evaluation
        mean_score = np.mean(scores)
        logger.info(f"\nMean Spearman correlation: {mean_score:.4f}")
        
        # Multi-target prediction for competition metric
        all_preds = []
        for i, target_name in enumerate(target_cols):
            train_data = lgb.Dataset(X_train, label=y_train[:, i])
            model = lgb.train(
                {'objective': 'regression', 'verbose': -1},
                train_data,
                num_boost_round=50
            )
            pred = model.predict(X_test)
            all_preds.append(pred)
        
        # Stack predictions
        stacked_preds = np.column_stack(all_preds)
        
        # Calculate Sharpe-like score (simplified)
        correlations = []
        for i in range(len(target_cols)):
            corr = spearmanr(y_test[:, i], stacked_preds[:, i])[0]
            if not np.isnan(corr):
                correlations.append(corr)
        
        if correlations:
            sharpe_like = np.mean(correlations) / np.std(correlations) if np.std(correlations) > 0 else np.mean(correlations)
            logger.info(f"Sharpe-like score: {sharpe_like:.4f}")
        
        logger.info("🎉 Minimal experiment completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"Success: {success}")