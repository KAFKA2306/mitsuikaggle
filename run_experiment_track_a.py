#!/usr/bin/env python3
"""
Run Experiment Track A: Multi-Target Learning Methods

This script implements the first experiment from our systematic research plan.
Starting with Independent Models baseline, then neural approaches if PyTorch is available.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.data.data_loader import MitsuiDataLoader
from src.features.feature_engineering import AdvancedFeatureEngineer
from src.evaluation.metrics import calculate_sharpe_like_score
from src.utils.experiment_manager import IntelligentExperimentRunner, create_experiment_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndependentModelsBaseline:
    """Independent LightGBM models for each target - our baseline approach."""
    
    def __init__(self, model_params: dict = None, random_state: int = 42):
        self.model_params = model_params or self._get_default_params()
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
    
    def _get_default_params(self) -> dict:
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
    
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: list) -> dict:
        """Train independent models for each target."""
        import lightgbm as lgb
        from scipy.stats import spearmanr
        
        logger.info(f"Training {len(target_names)} independent LightGBM models...")
        
        results = {
            'model_type': 'independent_models',
            'n_targets': len(target_names),
            'individual_scores': {},
            'feature_importance': {},
            'successful_targets': []
        }
        
        for i, target_name in enumerate(target_names):
            try:
                y_target = y[:, i]
                
                # Skip targets with too many missing values
                valid_mask = ~np.isnan(y_target)
                if valid_mask.sum() < 100:
                    logger.warning(f"Skipping {target_name} - insufficient valid data ({valid_mask.sum()} samples)")
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
                    num_boost_round=300,  # Reduced for speed
                    callbacks=[
                        lgb.early_stopping(30),
                        lgb.log_evaluation(0)
                    ]
                )
                
                # Evaluate
                val_pred = model.predict(X_val, num_iteration=model.best_iteration)
                score = spearmanr(y_val, val_pred)[0]
                if np.isnan(score):
                    score = 0.0
                
                # Store results
                self.models[target_name] = model
                results['individual_scores'][target_name] = score
                results['feature_importance'][target_name] = model.feature_importance(importance_type='gain').tolist()
                results['successful_targets'].append(target_name)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Completed {i + 1}/{len(target_names)} models, latest score: {score:.4f}")
                    
            except Exception as e:
                logger.error(f"Failed to train model for {target_name}: {e}")
                continue
        
        logger.info(f"Successfully trained {len(results['successful_targets'])}/{len(target_names)} models")
        return results
    
    def predict(self, X: np.ndarray, target_names: list) -> np.ndarray:
        """Make predictions for all targets."""
        predictions = []
        
        for target_name in target_names:
            if target_name in self.models:
                pred = self.models[target_name].predict(X, num_iteration=self.models[target_name].best_iteration)
            else:
                pred = np.zeros(len(X))
            predictions.append(pred)
        
        return np.column_stack(predictions)


def setup_experiment_data(sample_size: int = 1000, n_targets: int = 10):
    """Setup data for experiments with reduced size for testing."""
    logger.info(f"Setting up experiment data (sample_size={sample_size}, n_targets={n_targets})...")
    
    # Load data
    data_loader = MitsuiDataLoader()
    
    # Get basic data first
    train_data = data_loader.load_train_data()
    train_labels = data_loader.load_train_labels()
    
    # Use sample for speed
    train_sample = train_data.head(sample_size)
    labels_sample = train_labels.head(sample_size)
    
    # Basic feature engineering
    feature_engineer = AdvancedFeatureEngineer(
        technical_indicators=True,
        cross_asset_features=False,  # Disable for speed
        regime_features=False,
        economic_features=True,
        lag_features=[1, 2, 3],
        rolling_windows=[5, 10],
        correlation_window=5
    )
    
    # Merge data
    merged_data = train_sample.merge(labels_sample, on='date_id', how='inner')
    
    # Get target columns (limit to n_targets)
    all_target_cols = [col for col in labels_sample.columns if col.startswith('target_')]
    target_cols = all_target_cols[:n_targets]
    
    logger.info(f"Using {len(target_cols)} targets for experiment")
    
    # Create features
    features = feature_engineer.create_features(merged_data)
    
    # Prepare arrays
    feature_cols = [col for col in features.columns if col not in target_cols + ['date_id']]
    X = features[feature_cols].fillna(0).values.astype(np.float32)
    y = merged_data[target_cols].fillna(0).values.astype(np.float32)
    
    logger.info(f"Data prepared: X={X.shape}, y={y.shape}")
    return X, y, target_cols, feature_cols


def run_independent_models_experiment():
    """Run the independent models baseline experiment."""
    logger.info("Starting Independent Models Baseline Experiment...")
    
    # Setup data
    X, y, target_names, feature_names = setup_experiment_data(sample_size=800, n_targets=8)
    
    # Initialize experiment manager
    experiment_manager = IntelligentExperimentRunner()
    
    def objective_func(config):
        """Objective function for independent models."""
        approach = IndependentModelsBaseline(
            model_params=config.model_params,
            random_state=42
        )
        
        # Train models
        training_results = approach.fit(X, y, target_names)
        
        # Make predictions for competition metric
        predictions = approach.predict(X, target_names)
        
        # Calculate overall Sharpe-like score
        sharpe_score = calculate_sharpe_like_score(y, predictions)
        
        # Calculate additional metrics
        individual_scores = list(training_results['individual_scores'].values())
        mean_individual_score = np.mean(individual_scores) if individual_scores else 0.0
        std_individual_score = np.std(individual_scores) if individual_scores else 0.0
        
        return {
            'metrics': {
                'sharpe_like_score': sharpe_score,
                'mean_individual_spearman': mean_individual_score,
                'std_individual_spearman': std_individual_score,
                'n_successful_targets': len(training_results['successful_targets']),
                'success_rate': len(training_results['successful_targets']) / len(target_names)
            },
            'artifacts': {
                'individual_scores': training_results['individual_scores'],
                'successful_targets': training_results['successful_targets'],
                'feature_importance': training_results.get('feature_importance', {}),
                'training_summary': training_results
            },
            'memory_usage': 0  # Placeholder
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
        notes="Baseline approach: separate LightGBM model for each target"
    )
    
    # Run experiment
    result = experiment_manager.run_experiment(config, objective_func)
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("INDEPENDENT MODELS EXPERIMENT RESULTS")
    logger.info("="*60)
    logger.info(f"Experiment ID: {result.experiment_id}")
    logger.info(f"Status: {result.status}")
    
    if result.status == 'completed':
        metrics = result.performance_metrics
        logger.info(f"Sharpe-like Score: {metrics.get('sharpe_like_score', 0):.4f}")
        logger.info(f"Mean Individual Spearman: {metrics.get('mean_individual_spearman', 0):.4f}")
        logger.info(f"Std Individual Spearman: {metrics.get('std_individual_spearman', 0):.4f}")
        logger.info(f"Successful Targets: {metrics.get('n_successful_targets', 0)}/{len(target_names)}")
        logger.info(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
        
        # Show best performing targets
        if 'individual_scores' in result.model_artifacts:
            individual_scores = result.model_artifacts['individual_scores']
            best_targets = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("\nTop 5 Individual Target Results:")
            for target, score in best_targets:
                logger.info(f"  {target}: {score:.4f}")
    
    else:
        logger.error(f"Experiment failed: {result.error_message}")
    
    # Generate and save report
    report_path = experiment_manager.save_experiment_report()
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    return result


def main():
    """Main function to run Experiment Track A."""
    logger.info("="*60)
    logger.info("EXPERIMENT TRACK A: MULTI-TARGET LEARNING METHODS")
    logger.info("="*60)
    logger.info("Research Question: Which multi-target architecture best captures cross-asset dependencies?")
    logger.info("")
    
    try:
        # Run Independent Models baseline
        logger.info("Phase 1: Independent Models Baseline")
        result = run_independent_models_experiment()
        
        # Check PyTorch availability for neural approaches
        try:
            import torch
            logger.info(f"\nPyTorch available (version {torch.__version__})")
            logger.info("Neural network experiments will be available in future runs")
        except ImportError:
            logger.info("\nPyTorch not available - skipping neural network experiments")
            logger.info("To enable neural approaches, install: pip install torch")
        
        logger.info("\nExperiment Track A Phase 1 completed!")
        logger.info("Next steps: Implement Shared-Bottom and Multi-Task GNN approaches")
        
        return result
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    result = main()