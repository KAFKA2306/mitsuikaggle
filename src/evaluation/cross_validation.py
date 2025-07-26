"""
Time Series Cross-Validation for Mitsui Commodity Prediction Challenge.

Implements competition-specific CV strategies that prevent data leakage
and focus on stability evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional, Dict, Any
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)


class TimeSeriesSplit(BaseCrossValidator):
    """
    Time Series cross-validator with expanding window.
    
    Ensures no future data leakage and maintains temporal order.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        initial_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = True
    ):
        """
        Initialize Time Series Cross-Validator.
        
        Args:
            n_splits: Number of splits
            initial_train_size: Initial training size (if None, use 1/n_splits of data)
            test_size: Test size for each split (if None, use equal splits)
            gap: Gap between train and test sets
            expanding_window: If True, use expanding window; if False, use sliding window
        """
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
    
    def split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target values (optional)
            groups: Group labels (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Determine test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # Determine initial training size
        if self.initial_train_size is None:
            initial_train_size = n_samples // (self.n_splits + 1)
        else:
            initial_train_size = self.initial_train_size
        
        logger.info(f"Time Series CV: {self.n_splits} splits, test_size={test_size}, "
                   f"initial_train_size={initial_train_size}, gap={self.gap}")
        
        for i in range(self.n_splits):
            # Calculate split boundaries
            if self.expanding_window:
                # Expanding window: training set grows with each split
                train_start = 0
                train_end = initial_train_size + i * test_size
            else:
                # Sliding window: training set size remains constant
                train_start = i * test_size
                train_end = initial_train_size + i * test_size
            
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                break
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            logger.info(f"Split {i+1}: train=[{train_start}:{train_end}], "
                       f"test=[{test_start}:{test_end}]")
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self, 
        X: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None, 
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


class StabilityFocusedCV:
    """
    Cross-validation framework focused on stability evaluation.
    
    Implements multiple validation strategies and combines them
    to evaluate model stability across different scenarios.
    """
    
    def __init__(
        self,
        primary_cv: BaseCrossValidator,
        stability_metrics: List[str] = None,
        min_improvement_threshold: float = 0.01,
        stability_weight: float = 0.3
    ):
        """
        Initialize Stability-Focused Cross-Validator.
        
        Args:
            primary_cv: Primary cross-validation strategy
            stability_metrics: List of stability metrics to track
            min_improvement_threshold: Minimum improvement to consider significant
            stability_weight: Weight given to stability vs accuracy (0-1)
        """
        self.primary_cv = primary_cv
        self.stability_metrics = stability_metrics or [
            'std_spearman_correlation',
            'mean_temporal_volatility',
            'prediction_consistency'
        ]
        self.min_improvement_threshold = min_improvement_threshold
        self.stability_weight = stability_weight
        
        # Results storage
        self.cv_results = []
        self.stability_scores = []
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring_func: callable,
        fit_params: Dict = None
    ) -> Dict[str, Any]:
        """
        Evaluate model using stability-focused cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            scoring_func: Function to calculate scores
            fit_params: Additional parameters for model fitting
            
        Returns:
            Dictionary containing evaluation results
        """
        if fit_params is None:
            fit_params = {}
        
        fold_scores = []
        fold_predictions = []
        fold_stability_metrics = []
        
        logger.info(f"Starting {self.primary_cv.get_n_splits()} fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(self.primary_cv.split(X, y)):
            logger.info(f"Processing fold {fold + 1}...")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train, **fit_params)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate scores
            scores = scoring_func(y_val, y_pred)
            fold_scores.append(scores)
            fold_predictions.append(y_pred)
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(y_val, y_pred)
            fold_stability_metrics.append(stability_metrics)
            
            logger.info(f"Fold {fold + 1} - Primary score: {scores.get('sharpe_like_score', 'N/A'):.4f}")
        
        # Aggregate results
        results = self._aggregate_cv_results(
            fold_scores, 
            fold_predictions, 
            fold_stability_metrics
        )
        
        # Store results
        self.cv_results.append(results)
        
        return results
    
    def _calculate_stability_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate stability-specific metrics."""
        from .metrics import calculate_stability_metrics
        
        stability_metrics = calculate_stability_metrics(y_pred)
        
        # Add prediction consistency metric
        if y_pred.ndim > 1:
            # Measure how consistent predictions are across targets
            pred_correlations = np.corrcoef(y_pred.T)
            consistency = np.mean(pred_correlations[np.triu_indices_from(pred_correlations, k=1)])
            stability_metrics['prediction_consistency'] = consistency
        
        return stability_metrics
    
    def _aggregate_cv_results(
        self,
        fold_scores: List[Dict],
        fold_predictions: List[np.ndarray],
        fold_stability_metrics: List[Dict]
    ) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        
        # Aggregate primary scores
        score_aggregates = {}
        for metric in fold_scores[0].keys():
            values = [scores[metric] for scores in fold_scores]
            score_aggregates[f'mean_{metric}'] = np.mean(values)
            score_aggregates[f'std_{metric}'] = np.std(values)
            score_aggregates[f'min_{metric}'] = np.min(values)
            score_aggregates[f'max_{metric}'] = np.max(values)
        
        # Aggregate stability metrics
        stability_aggregates = {}
        if fold_stability_metrics:
            for metric in fold_stability_metrics[0].keys():
                values = [stability[metric] for stability in fold_stability_metrics]
                stability_aggregates[f'mean_{metric}'] = np.mean(values)
                stability_aggregates[f'std_{metric}'] = np.std(values)
        
        # Calculate combined stability score
        combined_score = self._calculate_combined_score(
            score_aggregates, 
            stability_aggregates
        )
        
        results = {
            'scores': score_aggregates,
            'stability': stability_aggregates,
            'combined_score': combined_score,
            'fold_predictions': fold_predictions,
            'n_folds': len(fold_scores)
        }
        
        return results
    
    def _calculate_combined_score(
        self,
        score_aggregates: Dict[str, float],
        stability_aggregates: Dict[str, float]
    ) -> float:
        """Calculate combined accuracy + stability score."""
        
        # Primary score (accuracy)
        primary_score = score_aggregates.get('mean_sharpe_like_score', 0.0)
        
        # Stability penalty (higher volatility = lower score)
        stability_penalty = 0.0
        
        if 'mean_std_spearman_correlation' in stability_aggregates:
            # Penalize high standard deviation in correlations
            stability_penalty += stability_aggregates['mean_std_spearman_correlation']
        
        if 'mean_mean_temporal_volatility' in stability_aggregates:
            # Penalize high temporal volatility
            stability_penalty += stability_aggregates['mean_mean_temporal_volatility'] * 0.1
        
        # Combine accuracy and stability
        combined_score = (
            (1 - self.stability_weight) * primary_score - 
            self.stability_weight * stability_penalty
        )
        
        return combined_score
    
    def get_best_model_params(self) -> Dict[str, Any]:
        """Get parameters of the best performing model."""
        if not self.cv_results:
            raise ValueError("No cross-validation results available")
        
        best_result = max(self.cv_results, key=lambda x: x['combined_score'])
        return best_result
    
    def generate_cv_report(self) -> pd.DataFrame:
        """Generate comprehensive cross-validation report."""
        if not self.cv_results:
            return pd.DataFrame()
        
        report_data = []
        for i, result in enumerate(self.cv_results):
            row = {'model_id': i}
            row.update(result['scores'])
            row.update(result['stability'])
            row['combined_score'] = result['combined_score']
            report_data.append(row)
        
        return pd.DataFrame(report_data)


class WalkForwardCV:
    """
    Walk-forward cross-validation for time series.
    
    Simulates real-world deployment where models are retrained
    periodically and evaluated on subsequent periods.
    """
    
    def __init__(
        self,
        initial_train_size: int,
        step_size: int,
        horizon: int,
        retrain_frequency: int = 1
    ):
        """
        Initialize Walk-Forward Cross-Validator.
        
        Args:
            initial_train_size: Initial training window size
            step_size: Step size for moving forward
            horizon: Prediction horizon
            retrain_frequency: How often to retrain (1 = every step)
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.horizon = horizon
        self.retrain_frequency = retrain_frequency
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Features
            y: Targets (optional)
            
        Yields:
            Tuple of (train_indices, test_indices, step_number)
        """
        n_samples = len(X)
        step = 0
        
        train_start = 0
        train_end = self.initial_train_size
        
        while train_end + self.horizon <= n_samples:
            test_start = train_end
            test_end = min(test_start + self.horizon, n_samples)
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices, step
            
            # Move forward
            step += 1
            train_end += self.step_size
            
            # Expand training window or slide it
            if step % self.retrain_frequency == 0:
                # Keep expanding training window
                pass
            else:
                # Slide training window
                train_start += self.step_size
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring_func: callable,
        fit_params: Dict = None
    ) -> Dict[str, Any]:
        """
        Evaluate model using walk-forward validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            scoring_func: Scoring function
            fit_params: Model fit parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        if fit_params is None:
            fit_params = {}
        
        step_scores = []
        step_predictions = []
        
        logger.info("Starting walk-forward validation...")
        
        for train_idx, test_idx, step in self.split(X, y):
            logger.info(f"Step {step + 1}: train size={len(train_idx)}, test size={len(test_idx)}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model (only if retrain step)
            if step % self.retrain_frequency == 0:
                model.fit(X_train, y_train, **fit_params)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate scores
            scores = scoring_func(y_test, y_pred)
            step_scores.append(scores)
            step_predictions.append(y_pred)
            
            logger.info(f"Step {step + 1} score: {scores.get('sharpe_like_score', 'N/A'):.4f}")
        
        # Aggregate results
        results = self._aggregate_walk_forward_results(step_scores, step_predictions)
        
        return results
    
    def _aggregate_walk_forward_results(
        self,
        step_scores: List[Dict],
        step_predictions: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Aggregate walk-forward results."""
        
        # Calculate score statistics
        score_stats = {}
        for metric in step_scores[0].keys():
            values = [scores[metric] for scores in step_scores]
            score_stats[f'mean_{metric}'] = np.mean(values)
            score_stats[f'std_{metric}'] = np.std(values)
            score_stats[f'trend_{metric}'] = self._calculate_trend(values)
        
        results = {
            'scores': score_stats,
            'step_predictions': step_predictions,
            'n_steps': len(step_scores)
        }
        
        return results
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values over time."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        return slope


def create_competition_cv(
    n_samples: int,
    cv_strategy: str = 'time_series',
    n_splits: int = 5,
    **kwargs
) -> BaseCrossValidator:
    """
    Create cross-validation strategy optimized for the competition.
    
    Args:
        n_samples: Number of samples in dataset
        cv_strategy: CV strategy ('time_series', 'walk_forward')
        n_splits: Number of splits
        **kwargs: Additional parameters for CV strategy
        
    Returns:
        Cross-validation object
    """
    if cv_strategy == 'time_series':
        return TimeSeriesSplit(
            n_splits=n_splits,
            initial_train_size=kwargs.get('initial_train_size', n_samples // (n_splits + 1)),
            test_size=kwargs.get('test_size', None),
            gap=kwargs.get('gap', 0),
            expanding_window=kwargs.get('expanding_window', True)
        )
    elif cv_strategy == 'walk_forward':
        return WalkForwardCV(
            initial_train_size=kwargs.get('initial_train_size', n_samples // 2),
            step_size=kwargs.get('step_size', n_samples // 10),
            horizon=kwargs.get('horizon', n_samples // 20),
            retrain_frequency=kwargs.get('retrain_frequency', 1)
        )
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")


def validate_cv_strategy(
    cv: BaseCrossValidator,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Validate cross-validation strategy.
    
    Args:
        cv: Cross-validation object
        X: Features
        y: Targets
        
    Returns:
        Validation results
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'split_info': []
    }
    
    splits = list(cv.split(X, y))
    
    # Check number of splits
    if len(splits) == 0:
        validation_results['is_valid'] = False
        validation_results['warnings'].append("No splits generated")
        return validation_results
    
    # Analyze each split
    for i, (train_idx, test_idx) in enumerate(splits):
        split_info = {
            'split_id': i,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_range': (train_idx.min(), train_idx.max()),
            'test_range': (test_idx.min(), test_idx.max()),
            'data_leakage': np.any(train_idx > test_idx.min())  # Check for leakage
        }
        validation_results['split_info'].append(split_info)
        
        # Check for data leakage
        if split_info['data_leakage']:
            validation_results['warnings'].append(
                f"Split {i}: Potential data leakage detected"
            )
        
        # Check for reasonable split sizes
        if split_info['train_size'] < 10:
            validation_results['warnings'].append(
                f"Split {i}: Very small training set ({split_info['train_size']} samples)"
            )
        
        if split_info['test_size'] < 5:
            validation_results['warnings'].append(
                f"Split {i}: Very small test set ({split_info['test_size']} samples)"
            )
    
    logger.info(f"CV validation completed: {len(splits)} splits, "
               f"{len(validation_results['warnings'])} warnings")
    
    return validation_results