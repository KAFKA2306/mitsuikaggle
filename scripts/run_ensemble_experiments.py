#!/usr/bin/env python3
"""
Run Experiment Track B: Advanced Ensemble Strategies

Simplified execution script for ensemble method comparisons.
Tests Classical, Hybrid, and Multi-Modal ensemble approaches.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ensemble_approaches():
    """Test ensemble approaches with minimal data."""
    logger.info("="*60)
    logger.info("EXPERIMENT TRACK B: ADVANCED ENSEMBLE STRATEGIES")
    logger.info("="*60)
    logger.info("Research Question: What ensemble combination maximizes stability + accuracy?")
    logger.info("")
    
    try:
        # Load minimal data
        logger.info("Loading sample data...")
        train_data = pd.read_csv('input/train.csv').head(300)  # Small sample
        train_labels = pd.read_csv('input/train_labels.csv').head(300)
        
        # Merge and prepare
        merged = train_data.merge(train_labels, on='date_id', how='inner')
        
        # Get features and targets
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:10]  # 10 features
        target_cols = [col for col in train_labels.columns if col.startswith('target_')][:4]  # 4 targets
        
        logger.info(f"Using {len(feature_cols)} features and {len(target_cols)} targets")
        
        # Prepare arrays
        X = merged[feature_cols].fillna(0).values.astype(np.float32)
        y = merged[target_cols].fillna(0).values.astype(np.float32)
        
        logger.info(f"Data prepared: X={X.shape}, y={y.shape}")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = {}
        
        # Test 1: Classical Ensemble (LightGBM + XGBoost)
        logger.info("\n" + "="*40)
        logger.info("Testing Classical Ensemble")
        logger.info("="*40)
        
        try:
            import lightgbm as lgb
            import xgboost as xgb
            from scipy.stats import spearmanr
            
            ensemble_preds = []
            ensemble_scores = []
            
            for i, target_name in enumerate(target_cols):
                y_target_train = y_train[:, i]
                y_target_test = y_test[:, i]
                
                # LightGBM
                lgb_train = lgb.Dataset(X_train, label=y_target_train)
                lgb_model = lgb.train(
                    {'objective': 'regression', 'verbose': -1},
                    lgb_train,
                    num_boost_round=50
                )
                pred_lgb = lgb_model.predict(X_test)
                
                # XGBoost
                xgb_train = xgb.DMatrix(X_train, label=y_target_train)
                xgb_model = xgb.train(
                    {'objective': 'reg:squarederror', 'verbosity': 0},
                    xgb_train,
                    num_boost_round=50
                )
                pred_xgb = xgb_model.predict(xgb.DMatrix(X_test))
                
                # Ensemble (equal weights)
                ensemble_pred = 0.5 * pred_lgb + 0.5 * pred_xgb
                ensemble_preds.append(ensemble_pred)
                
                # Evaluate
                lgb_score = spearmanr(y_target_test, pred_lgb)[0] if not np.isnan(spearmanr(y_target_test, pred_lgb)[0]) else 0.0
                xgb_score = spearmanr(y_target_test, pred_xgb)[0] if not np.isnan(spearmanr(y_target_test, pred_xgb)[0]) else 0.0
                ensemble_score = spearmanr(y_target_test, ensemble_pred)[0] if not np.isnan(spearmanr(y_target_test, ensemble_pred)[0]) else 0.0
                
                ensemble_scores.append(ensemble_score)
                
                logger.info(f"Target {i+1}: LGB={lgb_score:.4f}, XGB={xgb_score:.4f}, Ensemble={ensemble_score:.4f}")
            
            # Competition metric
            stacked_ensemble = np.column_stack(ensemble_preds)
            from src.evaluation.metrics import calculate_sharpe_like_score
            classical_sharpe = calculate_sharpe_like_score(y_test, stacked_ensemble)
            
            results['classical_ensemble'] = {
                'sharpe_like_score': classical_sharpe,
                'mean_spearman': np.mean(ensemble_scores),
                'individual_scores': ensemble_scores
            }
            
            logger.info(f"✓ Classical Ensemble: Sharpe-like={classical_sharpe:.4f}, Mean Spearman={np.mean(ensemble_scores):.4f}")
            
        except Exception as e:
            logger.error(f"✗ Classical ensemble failed: {e}")
            results['classical_ensemble'] = {'error': str(e)}
        
        # Test 2: Hybrid Linear + Neural Ensemble
        logger.info("\n" + "="*40)
        logger.info("Testing Hybrid Linear + Neural Ensemble")
        logger.info("="*40)
        
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            import torch
            import torch.nn as nn
            
            hybrid_preds = []
            hybrid_scores = []
            
            for i, target_name in enumerate(target_cols):
                y_target_train = y_train[:, i]
                y_target_test = y_test[:, i]
                
                # Scale data
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)
                y_train_scaled = scaler_y.fit_transform(y_target_train.reshape(-1, 1)).flatten()
                
                # Linear component
                linear_model = Ridge(alpha=1.0)
                linear_model.fit(X_train_scaled, y_train_scaled)
                linear_pred_scaled = linear_model.predict(X_test_scaled)
                linear_pred = scaler_y.inverse_transform(linear_pred_scaled.reshape(-1, 1)).flatten()
                
                # Neural component
                torch.manual_seed(42)
                neural_model = nn.Sequential(
                    nn.Linear(X_train.shape[1], 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                )
                
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(neural_model.parameters(), lr=0.01)
                
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.FloatTensor(y_train_scaled)
                
                # Quick training
                neural_model.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    output = neural_model(X_train_tensor).squeeze()
                    loss = criterion(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Neural predictions
                neural_model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test_scaled)
                    neural_pred_scaled = neural_model(X_test_tensor).squeeze().numpy()
                    neural_pred = scaler_y.inverse_transform(neural_pred_scaled.reshape(-1, 1)).flatten()
                
                # Hybrid ensemble
                hybrid_pred = 0.6 * linear_pred + 0.4 * neural_pred
                hybrid_preds.append(hybrid_pred)
                
                # Evaluate
                linear_score = spearmanr(y_target_test, linear_pred)[0] if not np.isnan(spearmanr(y_target_test, linear_pred)[0]) else 0.0
                neural_score = spearmanr(y_target_test, neural_pred)[0] if not np.isnan(spearmanr(y_target_test, neural_pred)[0]) else 0.0
                hybrid_score = spearmanr(y_target_test, hybrid_pred)[0] if not np.isnan(spearmanr(y_target_test, hybrid_pred)[0]) else 0.0
                
                hybrid_scores.append(hybrid_score)
                
                logger.info(f"Target {i+1}: Linear={linear_score:.4f}, Neural={neural_score:.4f}, Hybrid={hybrid_score:.4f}")
            
            # Competition metric
            stacked_hybrid = np.column_stack(hybrid_preds)
            hybrid_sharpe = calculate_sharpe_like_score(y_test, stacked_hybrid)
            
            results['hybrid_ensemble'] = {
                'sharpe_like_score': hybrid_sharpe,
                'mean_spearman': np.mean(hybrid_scores),
                'individual_scores': hybrid_scores
            }
            
            logger.info(f"✓ Hybrid Ensemble: Sharpe-like={hybrid_sharpe:.4f}, Mean Spearman={np.mean(hybrid_scores):.4f}")
            
        except Exception as e:
            logger.error(f"✗ Hybrid ensemble failed: {e}")
            results['hybrid_ensemble'] = {'error': str(e)}
        
        # Test 3: Multi-Model Voting Ensemble
        logger.info("\n" + "="*40)
        logger.info("Testing Multi-Model Voting Ensemble")
        logger.info("="*40)
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import ElasticNet
            
            voting_preds = []
            voting_scores = []
            
            for i, target_name in enumerate(target_cols):
                y_target_train = y_train[:, i]
                y_target_test = y_test[:, i]
                
                # Model 1: LightGBM
                lgb_train = lgb.Dataset(X_train, label=y_target_train)
                lgb_model = lgb.train(
                    {'objective': 'regression', 'verbose': -1},
                    lgb_train,
                    num_boost_round=40
                )
                pred_lgb = lgb_model.predict(X_test)
                
                # Model 2: Random Forest
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(X_train, y_target_train)
                pred_rf = rf_model.predict(X_test)
                
                # Model 3: ElasticNet
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                en_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                en_model.fit(X_train_scaled, y_target_train)
                pred_en = en_model.predict(X_test_scaled)
                
                # Voting ensemble (equal weights)
                voting_pred = (pred_lgb + pred_rf + pred_en) / 3
                voting_preds.append(voting_pred)
                
                # Evaluate
                lgb_score = spearmanr(y_target_test, pred_lgb)[0] if not np.isnan(spearmanr(y_target_test, pred_lgb)[0]) else 0.0
                rf_score = spearmanr(y_target_test, pred_rf)[0] if not np.isnan(spearmanr(y_target_test, pred_rf)[0]) else 0.0
                en_score = spearmanr(y_target_test, pred_en)[0] if not np.isnan(spearmanr(y_target_test, pred_en)[0]) else 0.0
                voting_score = spearmanr(y_target_test, voting_pred)[0] if not np.isnan(spearmanr(y_target_test, voting_pred)[0]) else 0.0
                
                voting_scores.append(voting_score)
                
                logger.info(f"Target {i+1}: LGB={lgb_score:.4f}, RF={rf_score:.4f}, EN={en_score:.4f}, Voting={voting_score:.4f}")
            
            # Competition metric
            stacked_voting = np.column_stack(voting_preds)
            voting_sharpe = calculate_sharpe_like_score(y_test, stacked_voting)
            
            results['voting_ensemble'] = {
                'sharpe_like_score': voting_sharpe,
                'mean_spearman': np.mean(voting_scores),
                'individual_scores': voting_scores
            }
            
            logger.info(f"✓ Voting Ensemble: Sharpe-like={voting_sharpe:.4f}, Mean Spearman={np.mean(voting_scores):.4f}")
            
        except Exception as e:
            logger.error(f"✗ Voting ensemble failed: {e}")
            results['voting_ensemble'] = {'error': str(e)}
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE EXPERIMENT RESULTS SUMMARY")
        logger.info("="*60)
        
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if successful_results:
            # Sort by Sharpe-like score
            sorted_results = sorted(
                successful_results.items(), 
                key=lambda x: x[1]['sharpe_like_score'], 
                reverse=True
            )
            
            logger.info("Ranking by Sharpe-like Score:")
            for i, (method, result) in enumerate(sorted_results):
                logger.info(f"{i+1}. {method.replace('_', ' ').title()}")
                logger.info(f"   Sharpe-like Score: {result['sharpe_like_score']:.4f}")
                logger.info(f"   Mean Spearman: {result['mean_spearman']:.4f}")
            
            # Best method analysis
            best_method, best_result = sorted_results[0]
            logger.info(f"\n🏆 Best Ensemble Method: {best_method.replace('_', ' ').title()}")
            logger.info(f"   Performance: {best_result['sharpe_like_score']:.4f} Sharpe-like score")
            
            if len(sorted_results) > 1:
                improvement = ((best_result['sharpe_like_score'] - sorted_results[1][1]['sharpe_like_score']) / 
                             sorted_results[1][1]['sharpe_like_score'] * 100)
                logger.info(f"   Improvement over second-best: {improvement:.1f}%")
        
        else:
            logger.warning("No ensemble experiments completed successfully")
        
        logger.info("\n🎉 Ensemble strategy experiments completed!")
        return results
        
    except Exception as e:
        logger.error(f"Ensemble experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main function."""
    logger.info("Starting Experiment Track B: Advanced Ensemble Strategies")
    
    # Add src to path
    sys.path.append('src')
    
    results = test_ensemble_approaches()
    
    logger.info("\nNext Steps:")
    logger.info("1. Experiment Track C: Advanced Feature Discovery")
    logger.info("2. Experiment Track D: Neural Architecture Search")
    logger.info("3. Experiment Track E: Competition-Specific Optimization")
    
    return results

if __name__ == "__main__":
    success = main()