"""
Enhanced LightGBM baseline model for Mitsui Commodity Prediction Challenge.

Implements competition-specific training with:
- Multi-target learning
- Competition metric optimization
- Advanced feature engineering
- Robust cross-validation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_loader import MitsuiDataLoader
from src.features.feature_engineering import AdvancedFeatureEngineer
from src.evaluation.metrics import CompetitionEvaluator, calculate_sharpe_like_score
from src.evaluation.cross_validation import TimeSeriesSplit, StabilityFocusedCV

logger = logging.getLogger(__name__)


class EnhancedLGBMModel:
    """
    Enhanced LightGBM model for commodity prediction.
    """
    
    def __init__(
        self,
        model_params: Dict = None,
        competition_metric_weight: float = 1.0,
        stability_weight: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize Enhanced LightGBM Model.
        
        Args:
            model_params: LightGBM parameters
            competition_metric_weight: Weight for competition metric in loss
            stability_weight: Weight for stability in evaluation
            random_state: Random seed
        """
        self.model_params = model_params or self._get_default_params()
        self.competition_metric_weight = competition_metric_weight
        self.stability_weight = stability_weight
        self.random_state = random_state
        
        # Model storage
        self.models = {}
        self.feature_names = None
        self.target_names = None
        
        # Results storage
        self.cv_results = {}
        self.feature_importance = {}
    
    def _get_default_params(self) -> Dict:
        """Get default LightGBM parameters optimized for competition."""
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
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def train_multi_target(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: List[str],
        cv_strategy: str = 'time_series',
        n_splits: int = 5,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Train multi-target LightGBM models.
        
        Args:
            X: Feature matrix
            y: Target matrix (n_samples, n_targets)
            target_names: List of target names
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV splits
            early_stopping_rounds: Early stopping rounds
            verbose: Verbose output
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training multi-target LightGBM with {y.shape[1]} targets...")
        
        # Store names
        self.target_names = target_names
        
        # Initialize CV strategy
        if cv_strategy == 'time_series':
            cv = TimeSeriesSplit(n_splits=n_splits, expanding_window=True)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        # Initialize evaluator
        evaluator = CompetitionEvaluator()
        
        # Results storage
        target_results = {}
        all_predictions = []
        all_actuals = []
        
        # Train models for each target
        for target_idx, target_name in enumerate(target_names):
            if verbose:
                logger.info(f"Training target {target_idx + 1}/{len(target_names)}: {target_name}")
            
            y_target = y[:, target_idx]
            
            # Skip targets with too many missing values
            missing_ratio = np.isnan(y_target).mean()
            if missing_ratio > 0.5:
                logger.warning(f"Skipping {target_name} - too many missing values ({missing_ratio:.2%})")
                continue
            
            # Train single target model
            model_results = self._train_single_target(
                X, y_target, target_name, cv, early_stopping_rounds, verbose
            )
            
            target_results[target_name] = model_results
            
            # Collect predictions for multi-target evaluation
            all_predictions.append(model_results['oof_predictions'])
            all_actuals.append(y_target)
        
        # Multi-target evaluation
        if all_predictions:
            # Stack predictions and actuals
            stacked_predictions = np.column_stack(all_predictions)
            stacked_actuals = np.column_stack(all_actuals)
            
            # Calculate competition metric
            competition_score = calculate_sharpe_like_score(stacked_actuals, stacked_predictions)
            
            logger.info(f"Multi-target Sharpe-like score: {competition_score:.4f}")
            
            # Store overall results
            self.cv_results = {
                'target_results': target_results,
                'multi_target_score': competition_score,
                'n_targets_trained': len(all_predictions),
                'overall_predictions': stacked_predictions,
                'overall_actuals': stacked_actuals
            }
        
        return self.cv_results
    
    def _train_single_target(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_name: str,
        cv,
        early_stopping_rounds: int,
        verbose: bool
    ) -> Dict:
        """Train model for single target."""
        
        # Remove NaN values
        valid_idx = ~np.isnan(y)
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        if len(y_valid) < 100:
            logger.warning(f"Too few valid samples for {target_name}: {len(y_valid)}")
            return {'error': 'insufficient_data'}
        
        # Cross-validation
        fold_scores = []
        fold_predictions = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_valid)):
            X_train, X_val = X_valid[train_idx], X_valid[val_idx]
            y_train, y_val = y_valid[train_idx], y_valid[val_idx]
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.model_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(0 if not verbose else 100)
                ]
            )
            
            # Predictions
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            fold_predictions.append((val_idx, val_pred))
            
            # Evaluation
            from scipy.stats import spearmanr
            fold_score = spearmanr(y_val, val_pred)[0]
            if np.isnan(fold_score):
                fold_score = 0.0
            
            fold_scores.append(fold_score)
            models.append(model)
            
            if verbose:
                logger.info(f"  Fold {fold + 1}: Spearman = {fold_score:.4f}")
        
        # Aggregate out-of-fold predictions
        oof_predictions = np.full(len(y_valid), np.nan)
        for val_idx, val_pred in fold_predictions:
            oof_predictions[val_idx] = val_pred
        
        # Full out-of-fold predictions (including NaN positions)
        full_oof = np.full(len(y), np.nan)
        full_oof[valid_idx] = oof_predictions
        
        # Calculate final score
        final_score = np.mean(fold_scores)
        score_std = np.std(fold_scores)
        
        # Feature importance (from last model)
        if models:
            feature_importance = models[-1].feature_importance(importance_type='gain')
            self.feature_importance[target_name] = feature_importance
        
        # Store models
        self.models[target_name] = models
        
        results = {
            'cv_score': final_score,
            'cv_std': score_std,
            'fold_scores': fold_scores,
            'oof_predictions': full_oof,
            'models': models,
            'n_valid_samples': len(y_valid)
        }
        
        return results
    
    def predict(self, X: np.ndarray, use_best_iteration: bool = True) -> np.ndarray:
        """
        Make predictions for all targets.
        
        Args:
            X: Feature matrix
            use_best_iteration: Whether to use best iteration from training
            
        Returns:
            Predictions array (n_samples, n_targets)
        """
        if not self.models:
            raise ValueError("No trained models found. Train the model first.")
        
        predictions = []
        
        for target_name in self.target_names:
            if target_name not in self.models:
                # Fill with zeros for missing targets
                target_pred = np.zeros(len(X))
            else:
                models = self.models[target_name]
                
                # Average predictions from all CV folds
                fold_predictions = []
                for model in models:
                    if use_best_iteration:
                        pred = model.predict(X, num_iteration=model.best_iteration)
                    else:
                        pred = model.predict(X)
                    fold_predictions.append(pred)
                
                target_pred = np.mean(fold_predictions, axis=0)
            
            predictions.append(target_pred)
        
        return np.column_stack(predictions)
    
    def get_feature_importance(self, target_name: str = None, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            target_name: Specific target name (if None, return average across all)
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            DataFrame with feature importance
        """
        if not self.feature_importance:
            logger.warning("No feature importance available")
            return pd.DataFrame()
        
        if target_name:
            if target_name not in self.feature_importance:
                logger.warning(f"No importance data for target {target_name}")
                return pd.DataFrame()
            
            importance = self.feature_importance[target_name]
            df = pd.DataFrame({
                'feature': range(len(importance)),
                'importance': importance
            })
            
            if self.feature_names:
                df['feature_name'] = self.feature_names
            
            return df.sort_values('importance', ascending=False)
        
        else:
            # Average importance across all targets
            all_importance = []
            for target, importance in self.feature_importance.items():
                all_importance.append(importance)
            
            avg_importance = np.mean(all_importance, axis=0)
            
            df = pd.DataFrame({
                'feature': range(len(avg_importance)),
                'importance': avg_importance
            })
            
            if self.feature_names:
                df['feature_name'] = self.feature_names
            
            return df.sort_values('importance', ascending=False)


def main():
    """Main training function."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Enhanced LightGBM Training...")
    
    try:
        # 1. Load data
        logger.info("Loading competition data...")
        data_loader = MitsuiDataLoader()
        data = data_loader.load_all_data()
        
        # 2. Feature engineering
        logger.info("Creating advanced features...")
        feature_engineer = AdvancedFeatureEngineer(
            technical_indicators=True,
            cross_asset_features=True,
            regime_features=True,
            economic_features=True,
            lag_features=[1, 2, 3, 5, 7, 10, 15, 20],
            rolling_windows=[5, 10, 20, 30],
            volatility_windows=[5, 10, 20],
            correlation_window=20
        )
        
        # Prepare data
        X, y = data_loader.prepare_model_data(
            drop_missing_targets=False,
            fill_missing_features='median'
        )
        
        # Create features
        feature_df = feature_engineer.create_features(
            pd.DataFrame(X), 
            target_cols=None
        )
        
        X_features = feature_df.values.astype(np.float32)
        
        # Store feature names
        feature_names = feature_df.columns.tolist()
        target_names = data_loader.target_columns
        
        logger.info(f"Feature matrix shape: {X_features.shape}")
        logger.info(f"Target matrix shape: {y.shape}")
        logger.info(f"Number of targets: {len(target_names)}")
        
        # 3. Train model
        logger.info("Training Enhanced LightGBM...")
        
        model = EnhancedLGBMModel(
            model_params={
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
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        )
        
        model.feature_names = feature_names
        
        # Train with limited targets for initial testing
        n_targets_to_train = min(10, len(target_names))
        logger.info(f"Training first {n_targets_to_train} targets for initial validation...")
        
        results = model.train_multi_target(
            X_features, 
            y[:, :n_targets_to_train],
            target_names[:n_targets_to_train],
            cv_strategy='time_series',
            n_splits=3,  # Reduced for faster initial testing
            early_stopping_rounds=50,
            verbose=True
        )
        
        # 4. Results summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*50)
        
        if 'multi_target_score' in results:
            logger.info(f"Multi-target Sharpe-like Score: {results['multi_target_score']:.4f}")
            logger.info(f"Targets Successfully Trained: {results['n_targets_trained']}")
        
        # Individual target results
        if 'target_results' in results:
            logger.info("\nIndividual Target Results:")
            for target_name, target_result in results['target_results'].items():
                if 'cv_score' in target_result:
                    logger.info(f"  {target_name}: CV Spearman = {target_result['cv_score']:.4f} ± {target_result['cv_std']:.4f}")
        
        logger.info("Training completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    results = main()