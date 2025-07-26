"""
Experiment Track A: Multi-Target Learning Methods

Implements and compares different multi-target architectures:
1. Independent Models: 424 separate LightGBM models
2. Shared-Bottom Multi-Task: Common features + target-specific heads
3. Multi-Task GNN: Graph neural network with cross-target relationships

This implements the first experiment from our systematic research plan.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
import os
from pathlib import Path
import uuid
from datetime import datetime

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

logger = logging.getLogger(__name__)


class IndependentModelsApproach:
    """Independent models for each target - baseline approach."""
    
    def __init__(self, model_params: Dict = None, random_state: int = 42):
        self.model_params = model_params or self._get_default_params()
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
    
    def _get_default_params(self) -> Dict:
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> Dict:
        """Train independent models for each target."""
        logger.info(f"Training {len(target_names)} independent LightGBM models...")
        
        results = {
            'model_type': 'independent_models',
            'n_targets': len(target_names),
            'individual_scores': {},
            'feature_importance': {}
        }
        
        for i, target_name in enumerate(target_names):
            y_target = y[:, i]
            
            # Skip targets with too many missing values
            valid_mask = ~np.isnan(y_target)
            if valid_mask.sum() < 100:
                logger.warning(f"Skipping {target_name} - insufficient valid data")
                continue
            
            X_valid = X[valid_mask]
            y_valid = y_target[valid_mask]
            
            # Simple train/validation split for efficiency
            split_idx = int(0.8 * len(X_valid))
            X_train, X_val = X_valid[:split_idx], X_valid[split_idx:]
            y_train, y_val = y_valid[:split_idx], y_valid[split_idx:]
            
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                self.model_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            # Evaluate
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            from scipy.stats import spearmanr
            score = spearmanr(y_val, val_pred)[0]
            if np.isnan(score):
                score = 0.0
            
            # Store results
            self.models[target_name] = model
            results['individual_scores'][target_name] = score
            results['feature_importance'][target_name] = model.feature_importance(importance_type='gain').tolist()
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(target_names)} models")
        
        return results
    
    def predict(self, X: np.ndarray, target_names: List[str]) -> np.ndarray:
        """Make predictions for all targets."""
        predictions = []
        
        for target_name in target_names:
            if target_name in self.models:
                pred = self.models[target_name].predict(X, num_iteration=self.models[target_name].best_iteration)
            else:
                pred = np.zeros(len(X))
            predictions.append(pred)
        
        return np.column_stack(predictions)


class SharedBottomMultiTask(nn.Module):
    """Shared-bottom multi-task neural network."""
    
    def __init__(self, input_dim: int, n_targets: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets
        
        # Shared bottom layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Target-specific heads
        self.target_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
            for _ in range(n_targets)
        ])
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Target-specific predictions
        predictions = []
        for head in self.target_heads:
            pred = head(shared_features)
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1)


class SharedBottomApproach:
    """Shared-bottom multi-task learning approach."""
    
    def __init__(self, hidden_dims: List[int] = [256, 128, 64], learning_rate: float = 0.001, random_state: int = 42):
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> Dict:
        """Train shared-bottom multi-task model."""
        logger.info("Training shared-bottom multi-task neural network...")
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Data preprocessing
        from sklearn.preprocessing import StandardScaler
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Handle missing targets
        y_processed = np.nan_to_num(y, nan=0.0)
        y_scaled = self.scaler_y.fit_transform(y_processed)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Initialize model
        self.model = SharedBottomMultiTask(
            input_dim=X.shape[1],
            n_targets=y.shape[1],
            hidden_dims=self.hidden_dims
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Simple train/validation split
        split_idx = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        train_losses = []
        val_losses = []
        
        for epoch in range(200):  # Max epochs
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            train_pred = self.model(X_train)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(X_val).numpy()
            y_val_orig = self.scaler_y.inverse_transform(y_val.numpy())
            val_pred_orig = self.scaler_y.inverse_transform(val_pred)
            
            # Calculate Sharpe-like score
            sharpe_score = calculate_sharpe_like_score(y_val_orig, val_pred_orig)
        
        results = {
            'model_type': 'shared_bottom',
            'n_targets': y.shape[1],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'sharpe_score': sharpe_score,
            'epochs_trained': len(train_losses)
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None or self.scaler_X is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).numpy()
            pred_orig = self.scaler_y.inverse_transform(pred_scaled)
        
        return pred_orig


class MultiTaskGNN(nn.Module):
    """Multi-task Graph Neural Network for cross-target relationships."""
    
    def __init__(self, input_dim: int, n_targets: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.hidden_dim = hidden_dim
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Graph attention layers for cross-target relationships
        self.gnn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Target-specific output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_targets)
        ])
        
        # Learnable target embeddings
        self.target_embeddings = nn.Parameter(torch.randn(n_targets, hidden_dim))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode input features
        encoded_features = self.feature_encoder(x)  # [batch_size, hidden_dim]
        
        # Create target-specific representations
        # Broadcast target embeddings for each sample
        target_features = self.target_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_targets, hidden_dim]
        
        # Add encoded features to each target
        target_features = target_features + encoded_features.unsqueeze(1)  # [batch_size, n_targets, hidden_dim]
        
        # Apply GNN layers for cross-target communication
        for gnn_layer in self.gnn_layers:
            attended_features, _ = gnn_layer(target_features, target_features, target_features)
            target_features = target_features + attended_features  # Residual connection
        
        # Generate predictions for each target
        predictions = []
        for i, output_layer in enumerate(self.output_layers):
            pred = output_layer(target_features[:, i, :])  # [batch_size, 1]
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1)  # [batch_size, n_targets]


class MultiTaskGNNApproach:
    """Multi-task Graph Neural Network approach."""
    
    def __init__(self, hidden_dim: int = 128, n_layers: int = 2, learning_rate: float = 0.001, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> Dict:
        """Train multi-task GNN model."""
        logger.info("Training multi-task Graph Neural Network...")
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Data preprocessing
        from sklearn.preprocessing import StandardScaler
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Handle missing targets
        y_processed = np.nan_to_num(y, nan=0.0)
        y_scaled = self.scaler_y.fit_transform(y_processed)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Initialize model
        self.model = MultiTaskGNN(
            input_dim=X.shape[1],
            n_targets=y.shape[1],
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Simple train/validation split
        split_idx = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        train_losses = []
        val_losses = []
        
        for epoch in range(200):  # Max epochs
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            train_pred = self.model(X_train)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(X_val).numpy()
            y_val_orig = self.scaler_y.inverse_transform(y_val.numpy())
            val_pred_orig = self.scaler_y.inverse_transform(val_pred)
            
            # Calculate Sharpe-like score
            sharpe_score = calculate_sharpe_like_score(y_val_orig, val_pred_orig)
        
        results = {
            'model_type': 'multi_task_gnn',
            'n_targets': y.shape[1],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'sharpe_score': sharpe_score,
            'epochs_trained': len(train_losses),
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None or self.scaler_X is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).numpy()
            pred_orig = self.scaler_y.inverse_transform(pred_scaled)
        
        return pred_orig


class MultiTargetExperimentRunner:
    """Experiment runner for multi-target learning methods comparison."""
    
    def __init__(self, experiment_manager: IntelligentExperimentRunner):
        self.experiment_manager = experiment_manager
        self.data_loader = None
        self.feature_engineer = None
        self.X = None
        self.y = None
        self.target_names = None
    
    def setup_data(self, sample_targets: int = 10) -> None:
        """Setup data for experiments."""
        logger.info("Setting up data for multi-target experiments...")
        
        # Load data
        self.data_loader = MitsuiDataLoader()
        
        # Prepare model data
        X, y = self.data_loader.prepare_model_data(
            drop_missing_targets=False,
            fill_missing_features='median'
        )
        
        # Use subset of targets for initial experiments
        self.target_names = self.data_loader.target_columns[:sample_targets]
        
        # Basic feature engineering for efficiency
        self.feature_engineer = AdvancedFeatureEngineer(
            technical_indicators=True,
            cross_asset_features=False,  # Disable for speed
            regime_features=False,
            economic_features=True,
            lag_features=[1, 2, 3],
            rolling_windows=[5, 10],
            correlation_window=10
        )
        
        # Create features
        train_data = self.data_loader.load_train_data()
        feature_df = self.feature_engineer.create_features(train_data.head(1000))  # Use subset for speed
        
        # Align data
        self.X = feature_df.fillna(0).values[:min(len(X), len(feature_df))]
        self.y = y[:len(self.X), :sample_targets]
        
        logger.info(f"Data setup complete: X={self.X.shape}, y={self.y.shape}")
    
    def run_independent_models_experiment(self) -> str:
        """Run independent models experiment."""
        
        def objective_func(config: ExperimentConfig) -> Dict[str, Any]:
            """Objective function for independent models."""
            approach = IndependentModelsApproach(
                model_params=config.model_params,
                random_state=42
            )
            
            # Train and evaluate
            results = approach.fit(self.X, self.y, self.target_names)
            
            # Make predictions for competition metric
            predictions = approach.predict(self.X, self.target_names)
            sharpe_score = calculate_sharpe_like_score(self.y, predictions)
            
            return {
                'metrics': {
                    'sharpe_like_score': sharpe_score,
                    'individual_scores_mean': np.mean(list(results['individual_scores'].values())),
                    'individual_scores_std': np.std(list(results['individual_scores'].values())),
                    'n_successful_targets': len(results['individual_scores'])
                },
                'artifacts': {
                    'individual_scores': results['individual_scores'],
                    'feature_importance': results['feature_importance']
                }
            }
        
        # Create experiment config
        config = create_experiment_config(
            name="Independent Models Baseline",
            model_type="independent_lgbm",
            model_params={
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            },
            notes="424 separate LightGBM models - baseline approach"
        )
        
        # Run experiment
        result = self.experiment_manager.run_experiment(config, objective_func)
        return result.experiment_id
    
    def run_shared_bottom_experiment(self) -> str:
        """Run shared-bottom multi-task experiment."""
        
        def objective_func(config: ExperimentConfig) -> Dict[str, Any]:
            """Objective function for shared-bottom approach."""
            approach = SharedBottomApproach(
                hidden_dims=config.model_params.get('hidden_dims', [256, 128, 64]),
                learning_rate=config.model_params.get('learning_rate', 0.001),
                random_state=42
            )
            
            # Train and evaluate
            results = approach.fit(self.X, self.y, self.target_names)
            
            # Make predictions for competition metric
            predictions = approach.predict(self.X)
            sharpe_score = calculate_sharpe_like_score(self.y, predictions)
            
            return {
                'metrics': {
                    'sharpe_like_score': sharpe_score,
                    'final_train_loss': results['final_train_loss'],
                    'final_val_loss': results['final_val_loss'],
                    'epochs_trained': results['epochs_trained']
                },
                'artifacts': {
                    'training_results': results
                }
            }
        
        # Create experiment config
        config = create_experiment_config(
            name="Shared-Bottom Multi-Task",
            model_type="shared_bottom_nn",
            model_params={
                'hidden_dims': [256, 128, 64],
                'learning_rate': 0.001
            },
            notes="Neural network with shared feature extraction and target-specific heads"
        )
        
        # Run experiment
        result = self.experiment_manager.run_experiment(config, objective_func)
        return result.experiment_id
    
    def run_multi_task_gnn_experiment(self) -> str:
        """Run multi-task GNN experiment."""
        
        def objective_func(config: ExperimentConfig) -> Dict[str, Any]:
            """Objective function for multi-task GNN approach."""
            approach = MultiTaskGNNApproach(
                hidden_dim=config.model_params.get('hidden_dim', 128),
                n_layers=config.model_params.get('n_layers', 2),
                learning_rate=config.model_params.get('learning_rate', 0.001),
                random_state=42
            )
            
            # Train and evaluate
            results = approach.fit(self.X, self.y, self.target_names)
            
            # Make predictions for competition metric
            predictions = approach.predict(self.X)
            sharpe_score = calculate_sharpe_like_score(self.y, predictions)
            
            return {
                'metrics': {
                    'sharpe_like_score': sharpe_score,
                    'final_train_loss': results['final_train_loss'],
                    'final_val_loss': results['final_val_loss'],
                    'epochs_trained': results['epochs_trained']
                },
                'artifacts': {
                    'training_results': results
                }
            }
        
        # Create experiment config
        config = create_experiment_config(
            name="Multi-Task Graph Neural Network",
            model_type="multi_task_gnn",
            model_params={
                'hidden_dim': 128,
                'n_layers': 2,
                'learning_rate': 0.001
            },
            notes="GNN with attention mechanism for cross-target relationships"
        )
        
        # Run experiment
        result = self.experiment_manager.run_experiment(config, objective_func)
        return result.experiment_id
    
    def run_all_experiments(self) -> Dict[str, str]:
        """Run all multi-target learning experiments."""
        logger.info("Starting Multi-Target Learning Methods Comparison...")
        
        # Setup data
        self.setup_data(sample_targets=10)  # Use 10 targets for initial comparison
        
        experiment_ids = {}
        
        # Run experiments
        logger.info("Running Independent Models experiment...")
        experiment_ids['independent_models'] = self.run_independent_models_experiment()
        
        logger.info("Running Shared-Bottom Multi-Task experiment...")
        experiment_ids['shared_bottom'] = self.run_shared_bottom_experiment()
        
        logger.info("Running Multi-Task GNN experiment...")
        experiment_ids['multi_task_gnn'] = self.run_multi_task_gnn_experiment()
        
        # Generate comparison report
        report = self.experiment_manager.generate_experiment_report()
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("="*60)
        
        # Display best results
        best_experiments = report['best_experiments']
        for i, exp in enumerate(best_experiments[:3]):
            logger.info(f"{i+1}. {exp['config_summary']['name']}")
            logger.info(f"   Model: {exp['model_type']}")
            logger.info(f"   Sharpe-like Score: {exp['performance'].get('sharpe_like_score', 0):.4f}")
            logger.info(f"   Experiment ID: {exp['experiment_id']}")
        
        # AI insights
        if report['ai_insights']:
            logger.info("\nAI Insights:")
            for insight in report['ai_insights']:
                logger.info(f"  • {insight}")
        
        return experiment_ids


def main():
    """Main function to run multi-target experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Initializing Multi-Target Learning Experiments...")
    
    # Initialize experiment manager
    experiment_manager = IntelligentExperimentRunner()
    
    # Create experiment runner
    runner = MultiTargetExperimentRunner(experiment_manager)
    
    # Run all experiments
    experiment_ids = runner.run_all_experiments()
    
    # Save final report
    report_path = experiment_manager.save_experiment_report()
    logger.info(f"Detailed experiment report saved to: {report_path}")
    
    logger.info("Multi-target learning experiments completed!")
    return experiment_ids


if __name__ == "__main__":
    results = main()