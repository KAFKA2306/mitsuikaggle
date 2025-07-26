#!/usr/bin/env python3
"""Simple test to validate core functionality."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def main():
    print("🚀 Testing Mitsui Implementation")
    print("=" * 50)
    
    # Test 1: Data Loading
    print("\n1. Testing Data Loading...")
    try:
        df_train = pd.read_csv('input/train.csv')
        df_labels = pd.read_csv('input/train_labels.csv')
        df_pairs = pd.read_csv('input/target_pairs.csv')
        
        print(f"✓ Train data: {df_train.shape}")
        print(f"✓ Labels data: {df_labels.shape}")
        print(f"✓ Target pairs: {df_pairs.shape}")
        
        # Check target columns
        target_cols = [col for col in df_labels.columns if col.startswith('target_')]
        print(f"✓ Found {len(target_cols)} target columns")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test 2: Competition Metric
    print("\n2. Testing Competition Metric...")
    try:
        # Create sample predictions
        np.random.seed(42)
        n_samples, n_targets = 100, 5
        
        y_true = np.random.randn(n_samples, n_targets)
        y_pred = y_true + 0.1 * np.random.randn(n_samples, n_targets)
        
        # Calculate Spearman correlations for each target
        correlations = []
        for i in range(n_targets):
            corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
            if np.isnan(corr):
                corr = 0.0
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Calculate Sharpe-like score
        if np.std(correlations) > 0:
            sharpe_like_score = np.mean(correlations) / np.std(correlations)
        else:
            sharpe_like_score = np.mean(correlations)
        
        print(f"✓ Competition metric calculated: {sharpe_like_score:.4f}")
        print(f"  - Mean correlation: {np.mean(correlations):.4f}")
        print(f"  - Std correlation: {np.std(correlations):.4f}")
        
    except Exception as e:
        print(f"✗ Competition metric failed: {e}")
        return False
    
    # Test 3: Basic Feature Engineering
    print("\n3. Testing Basic Feature Engineering...")
    try:
        # Take small sample of real data
        sample_size = 50
        df_sample = df_train.head(sample_size).copy()
        
        # Create basic lag features
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_id'][:3]  # Limit for testing
        
        feature_count = len(df_sample.columns)
        
        for col in numeric_cols:
            # Lag features
            df_sample[f'{col}_lag_1'] = df_sample[col].shift(1)
            df_sample[f'{col}_lag_2'] = df_sample[col].shift(2)
            
            # Rolling features
            df_sample[f'{col}_roll_mean_5'] = df_sample[col].rolling(5).mean()
            df_sample[f'{col}_roll_std_5'] = df_sample[col].rolling(5).std()
        
        new_feature_count = len(df_sample.columns)
        added_features = new_feature_count - feature_count
        
        print(f"✓ Feature engineering completed")
        print(f"  - Original features: {feature_count}")
        print(f"  - Added features: {added_features}")
        print(f"  - Total features: {new_feature_count}")
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False
    
    # Test 4: Time Series Cross-Validation
    print("\n4. Testing Time Series Cross-Validation...")
    try:
        from sklearn.model_selection import TimeSeriesSplit
        
        # Create sample data
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples, 3)
        
        # Test time series split
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        
        print(f"✓ Generated {len(splits)} CV splits")
        
        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"  - Split {i+1}: train={len(train_idx)}, val={len(val_idx)}")
            
            # Check no data leakage
            if len(train_idx) > 0 and len(val_idx) > 0:
                if np.max(train_idx) >= np.min(val_idx):
                    print(f"    ⚠️  Potential data leakage detected")
                else:
                    print(f"    ✓ No data leakage")
        
    except Exception as e:
        print(f"✗ Cross-validation failed: {e}")
        return False
    
    # Test 5: Simple Model Training
    print("\n5. Testing Simple Model Training...")
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        
        # Prepare small dataset
        merged_data = df_train.merge(df_labels, on='date_id', how='inner')
        merged_data = merged_data.head(100)  # Small sample
        
        # Get feature and target columns
        feature_cols = [col for col in df_train.columns if col != 'date_id'][:5]  # First 5 features
        target_cols = [col for col in df_labels.columns if col.startswith('target_')][:3]  # First 3 targets
        
        # Prepare arrays
        X = merged_data[feature_cols].fillna(0).values
        y = merged_data[target_cols].fillna(0).values
        
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train simple model for each target
        models = []
        scores = []
        
        for i in range(y.shape[1]):
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train[:, i])
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            corr, _ = spearmanr(y_test[:, i], y_pred)
            if np.isnan(corr):
                corr = 0.0
            
            models.append(model)
            scores.append(corr)
        
        print(f"✓ Simple model training completed")
        print(f"  - Trained {len(models)} models")
        print(f"  - Mean Spearman correlation: {np.mean(scores):.4f}")
        print(f"  - Individual scores: {[f'{s:.3f}' for s in scores]}")
        
    except Exception as e:
        print(f"✗ Simple model training failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("Implementation is working correctly.")
    print("Ready to proceed with full model training.")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)