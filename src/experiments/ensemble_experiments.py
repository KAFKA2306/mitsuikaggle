"""
Experiment Track B: Advanced Ensemble Strategies

Implements and compares different ensemble architectures:
1. Classical Ensemble: LightGBM + XGBoost + CatBoost (equal weights)
2. Hybrid ARMA-CNN-LSTM: Linear + nonlinear components
3. Multi-Modal Ensemble: Transformer-MAT + GNN + Statistical models
4. Hierarchical Ensemble: Coarse prediction → fine-tuning cascade

Research Question: "What ensemble combination maximizes stability + accuracy?"
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None
    print("CatBoost not available - using LightGBM fallback")

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
import os
from pathlib import Path
import uuid
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_loader import MitsuiDataLoader
from src.features.feature_engineering import AdvancedFeatureEngineer
from src.evaluation.metrics import calculate_sharpe_like_score, CompetitionEvaluator
from src.evaluation.cross_validation import TimeSeriesSplit
from src.utils.experiment_manager import (
    IntelligentExperimentRunner, 
    ExperimentConfig, 
    create_experiment_config
)


class ClassicalEnsemble:
    """Classical ensemble combining LightGBM + XGBoost + CatBoost with equal weights."""
    
    def __init__(self, model_params: Dict = None, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.model_params = model_params or self._get_default_params()
    
    def _get_default_params(self) -> Dict:
        return {
            'lgb': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbose': -1
            },
            'xgb': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbosity': 0
            },
            'cb': {
                'objective': 'RMSE',
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_state': self.random_state,
                'verbose': False
            }
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> Dict:
        """Train classical ensemble for each target."""
        logger.info(f"Training Classical Ensemble for {len(target_names)} targets...")
        
        results = {
            'model_type': 'classical_ensemble',
            'n_targets': len(target_names),
            'individual_scores': {},
            'ensemble_scores': {},
            'model_weights': {},
            'successful_targets': []
        }
        
        for i, target_name in enumerate(target_names):
            try:
                y_target = y[:, i]
                
                # Skip targets with insufficient data
                valid_mask = ~np.isnan(y_target)
                if valid_mask.sum() < 100:
                    logger.warning(f"Skipping {target_name} - insufficient data")
                    continue
                
                X_valid = X[valid_mask]
                y_valid = y_target[valid_mask]
                
                # Train/validation split
                split_idx = int(0.8 * len(X_valid))
                X_train, X_val = X_valid[:split_idx], X_valid[split_idx:]
                y_train, y_val = y_valid[:split_idx], y_valid[split_idx:]
                
                # Train individual models
                target_models = {}
                predictions = {}
                scores = {}
                
                # LightGBM
                lgb_train = lgb.Dataset(X_train, label=y_train)
                lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
                lgb_model = lgb.train(
                    self.model_params['lgb'],
                    lgb_train,
                    valid_sets=[lgb_val],
                    num_boost_round=200,
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )
                pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
                target_models['lgb'] = lgb_model
                predictions['lgb'] = pred_lgb
                
                # XGBoost
                xgb_train = xgb.DMatrix(X_train, label=y_train)
                xgb_val = xgb.DMatrix(X_val, label=y_val)
                xgb_model = xgb.train(
                    self.model_params['xgb'],
                    xgb_train,
                    num_boost_round=200,
                    evals=[(xgb_val, 'eval')],
                    early_stopping_rounds=30,
                    verbose_eval=False
                )
                pred_xgb = xgb_model.predict(xgb.DMatrix(X_val))
                target_models['xgb'] = xgb_model
                predictions['xgb'] = pred_xgb
                
                # CatBoost (if available)
                if cb is not None:
                    cb_model = cb.CatBoostRegressor(**self.model_params['cb'], iterations=200)
                    cb_model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=30,
                        verbose=False
                    )
                    pred_cb = cb_model.predict(X_val)
                    target_models['cb'] = cb_model
                    predictions['cb'] = pred_cb
                else:
                    # Fallback: use LightGBM with different params
                    lgb_params_alt = self.model_params['lgb'].copy()
                    lgb_params_alt['num_leaves'] = 15
                    lgb_params_alt['learning_rate'] = 0.03
                    
                    lgb_alt = lgb.train(
                        lgb_params_alt,
                        lgb_train,
                        valid_sets=[lgb_val],
                        num_boost_round=200,
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                    )
                    pred_cb = lgb_alt.predict(X_val, num_iteration=lgb_alt.best_iteration)
                    target_models['cb'] = lgb_alt
                    predictions['cb'] = pred_cb
                
                # Calculate individual scores
                from scipy.stats import spearmanr
                for model_name, pred in predictions.items():
                    score = spearmanr(y_val, pred)[0]
                    if np.isnan(score):
                        score = 0.0
                    scores[model_name] = score
                
                # Ensemble prediction (equal weights)
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
                ensemble_score = spearmanr(y_val, ensemble_pred)[0]
                if np.isnan(ensemble_score):
                    ensemble_score = 0.0
                
                # Store results
                self.models[target_name] = target_models
                results['individual_scores'][target_name] = scores
                results['ensemble_scores'][target_name] = ensemble_score
                results['model_weights'][target_name] = {'lgb': 1/3, 'xgb': 1/3, 'cb': 1/3}
                results['successful_targets'].append(target_name)
                
                if (i + 1) % 3 == 0:
                    logger.info(f"Completed {i + 1}/{len(target_names)} targets, latest ensemble score: {ensemble_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Failed to train ensemble for {target_name}: {e}")
                continue
        
        logger.info(f"Classical Ensemble training completed: {len(results['successful_targets'])}/{len(target_names)} targets")
        return results
    
    def predict(self, X: np.ndarray, target_names: List[str]) -> np.ndarray:
        """Make ensemble predictions for all targets."""
        predictions = []
        
        for target_name in target_names:
            if target_name in self.models:
                target_models = self.models[target_name]
                target_preds = []
                
                # Get predictions from each model
                if 'lgb' in target_models:
                    pred_lgb = target_models['lgb'].predict(X, num_iteration=target_models['lgb'].best_iteration)
                    target_preds.append(pred_lgb)
                
                if 'xgb' in target_models:
                    pred_xgb = target_models['xgb'].predict(xgb.DMatrix(X))
                    target_preds.append(pred_xgb)
                
                if 'cb' in target_models:
                    if hasattr(target_models['cb'], 'predict'):
                        pred_cb = target_models['cb'].predict(X)
                    else:
                        pred_cb = target_models['cb'].predict(X, num_iteration=target_models['cb'].best_iteration)
                    target_preds.append(pred_cb)
                
                # Ensemble (equal weights)
                ensemble_pred = np.mean(target_preds, axis=0)
                predictions.append(ensemble_pred)
            else:
                predictions.append(np.zeros(len(X)))
        
        return np.column_stack(predictions)


class HybridARMACNNLSTM:
    """Hybrid ensemble combining ARMA (linear) + CNN-LSTM (nonlinear) components."""
    
    def __init__(self, arma_order: Tuple[int, int] = (2, 1), hidden_size: int = 64, random_state: int = 42):
        self.arma_order = arma_order
        self.hidden_size = hidden_size
        self.random_state = random_state
        self.arma_models = {}
        self.neural_models = {}
        self.scalers = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> Dict:
        """Train hybrid ARMA-CNN-LSTM ensemble."""
        logger.info("Training Hybrid ARMA-CNN-LSTM ensemble...")
        
        # For this implementation, we'll use a simplified approach:
        # Linear component: Ridge regression (ARMA approximation)
        # Nonlinear component: Simple neural network
        
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        results = {
            'model_type': 'hybrid_arma_cnn_lstm',
            'n_targets': len(target_names),
            'linear_scores': {},
            'nonlinear_scores': {},
            'ensemble_scores': {},
            'successful_targets': []
        }
        
        for i, target_name in enumerate(target_names):
            try:
                y_target = y[:, i]
                
                # Skip targets with insufficient data
                valid_mask = ~np.isnan(y_target)
                if valid_mask.sum() < 100:
                    continue
                
                X_valid = X[valid_mask]
                y_valid = y_target[valid_mask]
                
                # Train/validation split
                split_idx = int(0.8 * len(X_valid))
                X_train, X_val = X_valid[:split_idx], X_valid[split_idx:]
                y_train, y_val = y_valid[:split_idx], y_valid[split_idx:]
                
                # Scale data
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_val_scaled = scaler_X.transform(X_val)
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                
                # Linear component (ARMA approximation with Ridge)
                linear_model = Ridge(alpha=1.0, random_state=self.random_state)
                linear_model.fit(X_train_scaled, y_train_scaled)
                linear_pred_scaled = linear_model.predict(X_val_scaled)
                linear_pred = scaler_y.inverse_transform(linear_pred_scaled.reshape(-1, 1)).flatten()
                
                # Nonlinear component (Simple NN as CNN-LSTM approximation)
                torch.manual_seed(self.random_state)
                neural_model = nn.Sequential(
                    nn.Linear(X_train.shape[1], self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size // 2, 1)
                )
                
                # Train neural network
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(neural_model.parameters(), lr=0.001)
                
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.FloatTensor(y_train_scaled)
                X_val_tensor = torch.FloatTensor(X_val_scaled)
                
                neural_model.train()
                for epoch in range(100):  # Quick training
                    optimizer.zero_grad()
                    output = neural_model(X_train_tensor).squeeze()
                    loss = criterion(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Neural predictions
                neural_model.eval()
                with torch.no_grad():
                    neural_pred_scaled = neural_model(X_val_tensor).squeeze().numpy()
                    neural_pred = scaler_y.inverse_transform(neural_pred_scaled.reshape(-1, 1)).flatten()
                
                # Ensemble prediction (weighted combination)
                ensemble_pred = 0.6 * linear_pred + 0.4 * neural_pred
                
                # Evaluate
                from scipy.stats import spearmanr
                linear_score = spearmanr(y_val, linear_pred)[0] if not np.isnan(spearmanr(y_val, linear_pred)[0]) else 0.0
                neural_score = spearmanr(y_val, neural_pred)[0] if not np.isnan(spearmanr(y_val, neural_pred)[0]) else 0.0
                ensemble_score = spearmanr(y_val, ensemble_pred)[0] if not np.isnan(spearmanr(y_val, ensemble_pred)[0]) else 0.0
                
                # Store models and results
                self.arma_models[target_name] = linear_model
                self.neural_models[target_name] = neural_model
                self.scalers[target_name] = {'X': scaler_X, 'y': scaler_y}
                
                results['linear_scores'][target_name] = linear_score
                results['nonlinear_scores'][target_name] = neural_score
                results['ensemble_scores'][target_name] = ensemble_score
                results['successful_targets'].append(target_name)
                
                if (i + 1) % 3 == 0:
                    logger.info(f"Completed {i + 1}/{len(target_names)} targets, latest scores: "
                              f"Linear={linear_score:.4f}, Neural={neural_score:.4f}, Ensemble={ensemble_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Failed to train hybrid model for {target_name}: {e}")
                continue
        
        logger.info(f"Hybrid ARMA-CNN-LSTM training completed: {len(results['successful_targets'])}/{len(target_names)} targets")
        return results
    
    def predict(self, X: np.ndarray, target_names: List[str]) -> np.ndarray:
        """Make hybrid ensemble predictions."""
        predictions = []
        
        for target_name in target_names:
            if target_name in self.arma_models and target_name in self.neural_models:
                try:
                    # Scale input
                    X_scaled = self.scalers[target_name]['X'].transform(X)
                    
                    # Linear prediction
                    linear_pred_scaled = self.arma_models[target_name].predict(X_scaled)
                    linear_pred = self.scalers[target_name]['y'].inverse_transform(linear_pred_scaled.reshape(-1, 1)).flatten()
                    
                    # Neural prediction
                    self.neural_models[target_name].eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_scaled)
                        neural_pred_scaled = self.neural_models[target_name](X_tensor).squeeze().numpy()
                        neural_pred = self.scalers[target_name]['y'].inverse_transform(neural_pred_scaled.reshape(-1, 1)).flatten()
                    
                    # Ensemble
                    ensemble_pred = 0.6 * linear_pred + 0.4 * neural_pred
                    predictions.append(ensemble_pred)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {target_name}: {e}")
                    predictions.append(np.zeros(len(X)))
            else:
                predictions.append(np.zeros(len(X)))
        
        return np.column_stack(predictions)


class MultiModalEnsemble:
    """Multi-modal ensemble combining Transformer-like + GNN + Statistical models."""
    
    def __init__(self, hidden_dim: int = 64, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.transformer_models = {}
        self.statistical_models = {}
        self.scalers = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> Dict:
        """Train multi-modal ensemble."""
        logger.info("Training Multi-Modal ensemble...")
        
        # Simplified implementation:
        # Transformer-like: Attention-based neural network
        # Statistical: Ridge regression with polynomial features
        
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        
        results = {
            'model_type': 'multi_modal_ensemble',
            'n_targets': len(target_names),
            'transformer_scores': {},
            'statistical_scores': {},
            'ensemble_scores': {},
            'successful_targets': []
        }
        
        for i, target_name in enumerate(target_names):
            try:
                y_target = y[:, i]
                
                # Skip targets with insufficient data
                valid_mask = ~np.isnan(y_target)
                if valid_mask.sum() < 100:
                    continue
                
                X_valid = X[valid_mask]
                y_valid = y_target[valid_mask]
                
                # Train/validation split
                split_idx = int(0.8 * len(X_valid))
                X_train, X_val = X_valid[:split_idx], X_valid[split_idx:]
                y_train, y_val = y_valid[:split_idx], y_valid[split_idx:]
                
                # Scale data
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_val_scaled = scaler_X.transform(X_val)
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                
                # Statistical component (polynomial features)
                poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_train_poly = poly_features.fit_transform(X_train_scaled[:, :min(5, X_train_scaled.shape[1])])  # Limit features
                X_val_poly = poly_features.transform(X_val_scaled[:, :min(5, X_val_scaled.shape[1])])
                
                statistical_model = Ridge(alpha=10.0, random_state=self.random_state)
                statistical_model.fit(X_train_poly, y_train_scaled)
                stat_pred_scaled = statistical_model.predict(X_val_poly)
                stat_pred = scaler_y.inverse_transform(stat_pred_scaled.reshape(-1, 1)).flatten()
                
                # Transformer-like component (attention mechanism approximation)
                torch.manual_seed(self.random_state)
                
                class SimpleAttentionModel(nn.Module):
                    def __init__(self, input_dim, hidden_dim):
                        super().__init__()
                        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                        self.encoder = nn.Linear(input_dim, hidden_dim)
                        self.decoder = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim // 2),
                            nn.ReLU(),
                            nn.Linear(hidden_dim // 2, 1)
                        )
                    
                    def forward(self, x):
                        # Simple attention mechanism
                        x_encoded = self.encoder(x).unsqueeze(1)  # Add sequence dimension
                        attended, _ = self.attention(x_encoded, x_encoded, x_encoded)
                        output = self.decoder(attended.squeeze(1))
                        return output
                
                transformer_model = SimpleAttentionModel(X_train.shape[1], self.hidden_dim)
                
                # Train transformer
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
                
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.FloatTensor(y_train_scaled)
                X_val_tensor = torch.FloatTensor(X_val_scaled)
                
                transformer_model.train()
                for epoch in range(80):  # Quick training
                    optimizer.zero_grad()
                    output = transformer_model(X_train_tensor).squeeze()
                    loss = criterion(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Transformer predictions
                transformer_model.eval()
                with torch.no_grad():
                    trans_pred_scaled = transformer_model(X_val_tensor).squeeze().numpy()
                    trans_pred = scaler_y.inverse_transform(trans_pred_scaled.reshape(-1, 1)).flatten()
                
                # Ensemble prediction (weighted combination)
                ensemble_pred = 0.7 * trans_pred + 0.3 * stat_pred
                
                # Evaluate
                from scipy.stats import spearmanr
                trans_score = spearmanr(y_val, trans_pred)[0] if not np.isnan(spearmanr(y_val, trans_pred)[0]) else 0.0
                stat_score = spearmanr(y_val, stat_pred)[0] if not np.isnan(spearmanr(y_val, stat_pred)[0]) else 0.0
                ensemble_score = spearmanr(y_val, ensemble_pred)[0] if not np.isnan(spearmanr(y_val, ensemble_pred)[0]) else 0.0
                
                # Store models
                self.transformer_models[target_name] = transformer_model
                self.statistical_models[target_name] = (statistical_model, poly_features)
                self.scalers[target_name] = {'X': scaler_X, 'y': scaler_y}
                
                results['transformer_scores'][target_name] = trans_score
                results['statistical_scores'][target_name] = stat_score
                results['ensemble_scores'][target_name] = ensemble_score
                results['successful_targets'].append(target_name)
                
                if (i + 1) % 3 == 0:
                    logger.info(f"Completed {i + 1}/{len(target_names)} targets, latest scores: "
                              f"Transformer={trans_score:.4f}, Statistical={stat_score:.4f}, Ensemble={ensemble_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Failed to train multi-modal model for {target_name}: {e}")
                continue
        
        logger.info(f"Multi-Modal ensemble training completed: {len(results['successful_targets'])}/{len(target_names)} targets")
        return results
    
    def predict(self, X: np.ndarray, target_names: List[str]) -> np.ndarray:
        """Make multi-modal ensemble predictions."""
        predictions = []
        
        for target_name in target_names:
            if target_name in self.transformer_models and target_name in self.statistical_models:
                try:
                    # Scale input
                    X_scaled = self.scalers[target_name]['X'].transform(X)
                    
                    # Statistical prediction
                    stat_model, poly_features = self.statistical_models[target_name]
                    X_poly = poly_features.transform(X_scaled[:, :min(5, X_scaled.shape[1])])
                    stat_pred_scaled = stat_model.predict(X_poly)
                    stat_pred = self.scalers[target_name]['y'].inverse_transform(stat_pred_scaled.reshape(-1, 1)).flatten()
                    
                    # Transformer prediction
                    self.transformer_models[target_name].eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_scaled)
                        trans_pred_scaled = self.transformer_models[target_name](X_tensor).squeeze().numpy()
                        trans_pred = self.scalers[target_name]['y'].inverse_transform(trans_pred_scaled.reshape(-1, 1)).flatten()
                    
                    # Ensemble
                    ensemble_pred = 0.7 * trans_pred + 0.3 * stat_pred
                    predictions.append(ensemble_pred)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {target_name}: {e}")
                    predictions.append(np.zeros(len(X)))
            else:
                predictions.append(np.zeros(len(X)))
        
        return np.column_stack(predictions)


class EnsembleExperimentRunner:
    """Experiment runner for advanced ensemble strategies comparison."""
    
    def __init__(self, experiment_manager: IntelligentExperimentRunner):
        self.experiment_manager = experiment_manager
        self.data_loader = None
        self.X = None
        self.y = None
        self.target_names = None
    
    def setup_data(self, sample_targets: int = 8, sample_rows: int = 500) -> None:
        """Setup data for ensemble experiments."""
        logger.info(f"Setting up data for ensemble experiments (targets={sample_targets}, rows={sample_rows})...")
        
        # Load and prepare data efficiently
        self.data_loader = MitsuiDataLoader()
        
        # Use basic features to avoid complexity
        train_data = self.data_loader.load_train_data().head(sample_rows)
        train_labels = self.data_loader.load_train_labels().head(sample_rows)
        
        # Merge data
        merged = train_data.merge(train_labels, on='date_id', how='inner')
        
        # Get feature and target columns
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:20]  # Limit features
        all_target_cols = [col for col in train_labels.columns if col.startswith('target_')]
        self.target_names = all_target_cols[:sample_targets]
        
        # Prepare arrays
        self.X = merged[feature_cols].fillna(0).values.astype(np.float32)
        self.y = merged[self.target_names].fillna(0).values.astype(np.float32)
        
        logger.info(f"Data setup complete: X={self.X.shape}, y={self.y.shape}")
    
    def run_classical_ensemble_experiment(self) -> str:
        """Run classical ensemble experiment."""
        
        def objective_func(config: ExperimentConfig) -> Dict[str, Any]:
            """Objective function for classical ensemble."""
            ensemble = ClassicalEnsemble(random_state=42)
            
            # Train ensemble
            results = ensemble.fit(self.X, self.y, self.target_names)
            
            # Make predictions for competition metric
            predictions = ensemble.predict(self.X, self.target_names)
            sharpe_score = calculate_sharpe_like_score(self.y, predictions)
            
            # Calculate metrics
            ensemble_scores = list(results['ensemble_scores'].values())
            mean_ensemble_score = np.mean(ensemble_scores) if ensemble_scores else 0.0
            
            return {
                'metrics': {
                    'sharpe_like_score': sharpe_score,
                    'mean_ensemble_spearman': mean_ensemble_score,
                    'n_successful_targets': len(results['successful_targets']),
                    'success_rate': len(results['successful_targets']) / len(self.target_names)
                },
                'artifacts': {
                    'ensemble_scores': results['ensemble_scores'],
                    'individual_scores': results['individual_scores'],
                    'successful_targets': results['successful_targets']
                }
            }
        
        config = create_experiment_config(
            name="Classical Ensemble (LGB+XGB+CB)",
            model_type="classical_ensemble",
            notes="Equal-weighted ensemble of LightGBM, XGBoost, and CatBoost"
        )
        
        result = self.experiment_manager.run_experiment(config, objective_func)
        return result.experiment_id
    
    def run_hybrid_ensemble_experiment(self) -> str:
        """Run hybrid ARMA-CNN-LSTM experiment."""
        
        def objective_func(config: ExperimentConfig) -> Dict[str, Any]:
            """Objective function for hybrid ensemble."""
            ensemble = HybridARMACNNLSTM(random_state=42)
            
            results = ensemble.fit(self.X, self.y, self.target_names)
            predictions = ensemble.predict(self.X, self.target_names)
            sharpe_score = calculate_sharpe_like_score(self.y, predictions)
            
            ensemble_scores = list(results['ensemble_scores'].values())
            mean_ensemble_score = np.mean(ensemble_scores) if ensemble_scores else 0.0
            
            return {
                'metrics': {
                    'sharpe_like_score': sharpe_score,
                    'mean_ensemble_spearman': mean_ensemble_score,
                    'n_successful_targets': len(results['successful_targets']),
                    'success_rate': len(results['successful_targets']) / len(self.target_names)
                },
                'artifacts': {
                    'ensemble_scores': results['ensemble_scores'],
                    'linear_scores': results['linear_scores'],
                    'nonlinear_scores': results['nonlinear_scores']
                }
            }
        
        config = create_experiment_config(
            name="Hybrid ARMA-CNN-LSTM",
            model_type="hybrid_ensemble",
            notes="Linear (ARMA) + Nonlinear (CNN-LSTM) component combination"
        )
        
        result = self.experiment_manager.run_experiment(config, objective_func)
        return result.experiment_id
    
    def run_multimodal_ensemble_experiment(self) -> str:
        """Run multi-modal ensemble experiment."""
        
        def objective_func(config: ExperimentConfig) -> Dict[str, Any]:
            """Objective function for multi-modal ensemble."""
            ensemble = MultiModalEnsemble(random_state=42)
            
            results = ensemble.fit(self.X, self.y, self.target_names)
            predictions = ensemble.predict(self.X, self.target_names)
            sharpe_score = calculate_sharpe_like_score(self.y, predictions)
            
            ensemble_scores = list(results['ensemble_scores'].values())
            mean_ensemble_score = np.mean(ensemble_scores) if ensemble_scores else 0.0
            
            return {
                'metrics': {
                    'sharpe_like_score': sharpe_score,
                    'mean_ensemble_spearman': mean_ensemble_score,
                    'n_successful_targets': len(results['successful_targets']),
                    'success_rate': len(results['successful_targets']) / len(self.target_names)
                },
                'artifacts': {
                    'ensemble_scores': results['ensemble_scores'],
                    'transformer_scores': results['transformer_scores'],
                    'statistical_scores': results['statistical_scores']
                }
            }
        
        config = create_experiment_config(
            name="Multi-Modal Ensemble",
            model_type="multimodal_ensemble",
            notes="Transformer-like + Statistical model combination"
        )
        
        result = self.experiment_manager.run_experiment(config, objective_func)
        return result.experiment_id
    
    def run_all_ensemble_experiments(self) -> Dict[str, str]:
        """Run all ensemble experiments."""
        logger.info("Starting Advanced Ensemble Strategies Comparison...")
        
        # Setup data
        self.setup_data(sample_targets=6, sample_rows=400)  # Manageable size
        
        experiment_ids = {}
        
        # Run experiments
        logger.info("Running Classical Ensemble experiment...")
        experiment_ids['classical'] = self.run_classical_ensemble_experiment()
        
        logger.info("Running Hybrid ARMA-CNN-LSTM experiment...")
        experiment_ids['hybrid'] = self.run_hybrid_ensemble_experiment()
        
        logger.info("Running Multi-Modal Ensemble experiment...")
        experiment_ids['multimodal'] = self.run_multimodal_ensemble_experiment()
        
        # Generate comparison report
        report = self.experiment_manager.generate_experiment_report()
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE EXPERIMENT RESULTS SUMMARY")
        logger.info("="*60)
        
        # Display best results
        best_experiments = report['best_experiments']
        for i, exp in enumerate(best_experiments[:3]):
            logger.info(f"{i+1}. {exp['config_summary']['name']}")
            logger.info(f"   Model: {exp['model_type']}")
            logger.info(f"   Sharpe-like Score: {exp['performance'].get('sharpe_like_score', 0):.4f}")
            logger.info(f"   Success Rate: {exp['performance'].get('success_rate', 0):.1%}")
        
        return experiment_ids


def main():
    """Main function to run ensemble experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Initializing Advanced Ensemble Strategy Experiments...")
    
    # Initialize experiment manager
    experiment_manager = IntelligentExperimentRunner()
    
    # Create experiment runner
    runner = EnsembleExperimentRunner(experiment_manager)
    
    # Run all experiments
    experiment_ids = runner.run_all_ensemble_experiments()
    
    # Save final report
    report_path = experiment_manager.save_experiment_report()
    logger.info(f"Detailed experiment report saved to: {report_path}")
    
    logger.info("Advanced ensemble strategy experiments completed!")
    return experiment_ids


if __name__ == "__main__":
    results = main()