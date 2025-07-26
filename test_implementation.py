#!/usr/bin/env python3
"""
Test script to validate our enhanced implementation works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.append('src')

from src.data.data_loader import MitsuiDataLoader
from src.evaluation.metrics import calculate_sharpe_like_score, CompetitionEvaluator
from src.evaluation.cross_validation import TimeSeriesSplit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        data_loader = MitsuiDataLoader()
        
        # Load individual files
        train_data = data_loader.load_train_data()
        train_labels = data_loader.load_train_labels()
        target_pairs = data_loader.load_target_pairs()
        
        logger.info(f"✓ Train data loaded: {train_data.shape}")
        logger.info(f"✓ Train labels loaded: {train_labels.shape}")
        logger.info(f"✓ Target pairs loaded: {target_pairs.shape}")
        
        # Test data preparation
        X, y = data_loader.prepare_model_data(
            drop_missing_targets=True,
            fill_missing_features='median'
        )
        
        logger.info(f"✓ Prepared data - X: {X.shape}, y: {y.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics functionality."""
    logger.info("Testing evaluation metrics...")
    
    try:
        # Create sample data
        np.random.seed(42)
        n_samples, n_targets = 100, 5
        
        y_true = np.random.randn(n_samples, n_targets)
        y_pred = y_true + 0.1 * np.random.randn(n_samples, n_targets)  # Add noise
        
        # Test competition metric
        score = calculate_sharpe_like_score(y_true, y_pred)
        logger.info(f"✓ Sharpe-like score calculated: {score:.4f}")
        
        # Test evaluator
        evaluator = CompetitionEvaluator()
        results = evaluator.evaluate(y_true, y_pred)
        
        logger.info(f"✓ Evaluation completed - {len(results)} metrics calculated")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Evaluation metrics failed: {e}")
        return False

def test_cross_validation():
    """Test cross-validation functionality."""
    logger.info("Testing cross-validation...")
    
    try:
        # Create sample data
        n_samples = 200
        X = np.random.randn(n_samples, 10)
        y = np.random.randn(n_samples, 3)
        
        # Test TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=3, expanding_window=True)
        
        splits = list(cv.split(X))
        logger.info(f"✓ Generated {len(splits)} CV splits")
        
        # Validate splits
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"  Split {i+1}: train={len(train_idx)}, val={len(val_idx)}")
            
            # Check no data leakage
            if np.any(train_idx > val_idx.min()):
                logger.warning(f"  Potential data leakage in split {i+1}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Cross-validation failed: {e}")
        return False

def test_basic_feature_engineering():
    """Test basic feature engineering."""
    logger.info("Testing feature engineering...")
    
    try:
        from src.features.feature_engineering import AdvancedFeatureEngineer
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'date_id': range(n_samples),
            'price1': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
            'price2': np.cumsum(np.random.randn(n_samples) * 0.01) + 50,
            'volume1': np.random.randint(1000, 10000, n_samples)
        })
        
        # Test feature engineering with limited features for speed
        engineer = AdvancedFeatureEngineer(
            technical_indicators=True,
            cross_asset_features=False,  # Disable for speed
            regime_features=True,
            economic_features=True,
            lag_features=[1, 2, 3],
            rolling_windows=[5, 10],
            correlation_window=10
        )
        
        features = engineer.create_features(df)
        logger.info(f"✓ Feature engineering completed: {features.shape}")
        logger.info(f"  Original features: {len(df.columns)}")
        logger.info(f"  Engineered features: {len(features.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        return False

def test_integration():
    """Test end-to-end integration with real data."""
    logger.info("Testing end-to-end integration...")
    
    try:
        # Load data
        data_loader = MitsuiDataLoader()
        train_data = data_loader.load_train_data()
        train_labels = data_loader.load_train_labels()
        
        # Take a small sample for testing
        sample_size = min(200, len(train_data))
        train_sample = train_data.head(sample_size)
        labels_sample = train_labels.head(sample_size)
        
        logger.info(f"Using sample of {sample_size} records for integration test")
        
        # Basic feature engineering (limited for speed)
        from src.features.feature_engineering import AdvancedFeatureEngineer
        
        engineer = AdvancedFeatureEngineer(
            technical_indicators=False,  # Disable for speed
            cross_asset_features=False,
            regime_features=False,
            economic_features=True,
            lag_features=[1, 2],
            rolling_windows=[5],
            correlation_window=5
        )
        
        # Merge data
        merged_data = train_sample.merge(labels_sample, on='date_id', how='inner')
        
        # Get target columns
        target_cols = [col for col in labels_sample.columns if col.startswith('target_')]
        target_cols = target_cols[:3]  # Use only first 3 targets for testing
        
        logger.info(f"Using {len(target_cols)} targets for testing")
        
        # Create features
        features = engineer.create_features(merged_data, target_cols=target_cols)
        
        # Prepare arrays
        feature_cols = [col for col in features.columns if col not in target_cols + ['date_id']]
        X = features[feature_cols].fillna(0).values
        y = merged_data[target_cols].fillna(0).values
        
        logger.info(f"Prepared data: X={X.shape}, y={y.shape}")
        
        # Test CV and evaluation
        cv = TimeSeriesSplit(n_splits=2, expanding_window=True)
        
        predictions = []
        actuals = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Simple linear model for testing
            from sklearn.linear_model import Ridge
            
            models = []
            val_preds = []
            
            for target_idx in range(y.shape[1]):
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_train, y_train[:, target_idx])
                pred = model.predict(X_val)
                val_preds.append(pred)
                models.append(model)
            
            val_pred_matrix = np.column_stack(val_preds)
            
            predictions.append(val_pred_matrix)
            actuals.append(y_val)
            
            logger.info(f"Fold {fold + 1} completed")
        
        # Calculate competition metric
        all_preds = np.vstack(predictions)
        all_actuals = np.vstack(actuals)
        
        score = calculate_sharpe_like_score(all_actuals, all_preds)
        logger.info(f"✓ Integration test completed - Sharpe-like score: {score:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Starting implementation validation tests...")
    logger.info("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Cross Validation", test_cross_validation),
        ("Feature Engineering", test_basic_feature_engineering),
        ("End-to-End Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        logger.info(f"{test_name:.<40} {status}")
        if passed_test:
            passed += 1
    
    logger.info("-"*60)
    logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! Implementation is ready.")
        return True
    else:
        logger.info("❌ Some tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)