#!/usr/bin/env python3
"""
Test experiment setup step by step to debug issues.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_step_by_step():
    """Test each step individually."""
    
    logger.info("Step 1: Testing imports...")
    try:
        sys.path.append('src')
        from src.data.data_loader import MitsuiDataLoader
        from src.features.feature_engineering import AdvancedFeatureEngineer
        from src.evaluation.metrics import calculate_sharpe_like_score
        logger.info("✓ All imports successful")
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False
    
    logger.info("Step 2: Testing data loading...")
    try:
        data_loader = MitsuiDataLoader()
        train_data = data_loader.load_train_data()
        train_labels = data_loader.load_train_labels()
        logger.info(f"✓ Data loaded: train={train_data.shape}, labels={train_labels.shape}")
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False
    
    logger.info("Step 3: Testing feature engineering...")
    try:
        feature_engineer = AdvancedFeatureEngineer(
            technical_indicators=False,  # Minimal for testing
            cross_asset_features=False,
            regime_features=False,
            economic_features=False,
            lag_features=[1],
            rolling_windows=[5],
            correlation_window=5
        )
        
        # Use small sample
        sample_data = train_data.head(50)
        sample_labels = train_labels.head(50)
        merged = sample_data.merge(sample_labels, on='date_id', how='inner')
        
        features = feature_engineer.create_features(merged)
        logger.info(f"✓ Features created: {features.shape}")
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("Step 4: Testing LightGBM...")
    try:
        import lightgbm as lgb
        from scipy.stats import spearmanr
        
        # Prepare simple data
        target_cols = [col for col in merged.columns if col.startswith('target_')][:2]
        feature_cols = [col for col in features.columns if col not in target_cols + ['date_id']][:10]
        
        X = features[feature_cols].fillna(0).values
        y = merged[target_cols[0]].fillna(0).values
        
        # Simple train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train simple model
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            {'objective': 'regression', 'verbose': -1},
            train_data,
            num_boost_round=10
        )
        
        # Predict and evaluate
        pred = model.predict(X_test)
        score = spearmanr(y_test, pred)[0]
        
        logger.info(f"✓ LightGBM test successful, Spearman correlation: {score:.4f}")
    except Exception as e:
        logger.error(f"✗ LightGBM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("Step 5: Testing experiment manager...")
    try:
        from src.utils.experiment_manager import IntelligentExperimentRunner, create_experiment_config
        
        manager = IntelligentExperimentRunner()
        config = create_experiment_config(
            name="Test Experiment",
            model_type="test",
            notes="Simple test"
        )
        
        logger.info(f"✓ Experiment manager created, config ID: {config.experiment_id}")
    except Exception as e:
        logger.error(f"✗ Experiment manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("🎉 All steps completed successfully!")
    return True

if __name__ == "__main__":
    success = test_step_by_step()
    print(f"Overall success: {success}")
    sys.exit(0 if success else 1)