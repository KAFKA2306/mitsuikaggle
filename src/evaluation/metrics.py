"""
Competition-specific evaluation metrics for Mitsui Commodity Prediction Challenge.

The primary metric is a Sharpe-like ratio of Spearman correlations:
Score = mean(spearman_correlations) / std(spearman_correlations)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Union, Tuple, Dict, List
import warnings


def calculate_spearman_correlation(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Handle edge cases
    if len(y_true) < 2:
        return 0.0
    
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    
    correlation, _ = spearmanr(y_true, y_pred)
    
    # Handle NaN correlations
    if np.isnan(correlation):
        return 0.0
    
    return correlation


def calculate_sharpe_like_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: int = 0
) -> float:
    """
    Calculate the competition's Sharpe-like evaluation metric.
    
    Score = mean(spearman_correlations) / std(spearman_correlations)
    
    Args:
        y_true: True values, shape (n_samples, n_targets)
        y_pred: Predicted values, shape (n_samples, n_targets)
        axis: Axis along which to calculate correlations (0 for time series)
        
    Returns:
        Sharpe-like score
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    # Handle 1D arrays
    if y_true.ndim == 1:
        correlation = calculate_spearman_correlation(y_true, y_pred)
        return correlation  # For single target, std is undefined
    
    # Calculate correlations for each target
    correlations = []
    n_targets = y_true.shape[1]
    
    for i in range(n_targets):
        corr = calculate_spearman_correlation(y_true[:, i], y_pred[:, i])
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # Remove NaN correlations
    valid_correlations = correlations[~np.isnan(correlations)]
    
    if len(valid_correlations) == 0:
        return 0.0
    
    if len(valid_correlations) == 1:
        return valid_correlations[0]
    
    mean_corr = np.mean(valid_correlations)
    std_corr = np.std(valid_correlations)
    
    # Avoid division by zero
    if std_corr == 0:
        return mean_corr
    
    sharpe_like_score = mean_corr / std_corr
    
    return sharpe_like_score


def evaluate_time_series_cv(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate model performance across time series cross-validation folds.
    
    Args:
        y_true_list: List of true values for each fold
        y_pred_list: List of predicted values for each fold
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true_list and y_pred_list must have same length")
    
    fold_scores = []
    fold_correlations = []
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        # Calculate Sharpe-like score for this fold
        score = calculate_sharpe_like_score(y_true, y_pred)
        fold_scores.append(score)
        
        # Calculate individual correlations for stability analysis
        if y_true.ndim > 1:
            correlations = []
            for i in range(y_true.shape[1]):
                corr = calculate_spearman_correlation(y_true[:, i], y_pred[:, i])
                correlations.append(corr)
            fold_correlations.extend(correlations)
        else:
            corr = calculate_spearman_correlation(y_true, y_pred)
            fold_correlations.append(corr)
    
    fold_scores = np.array(fold_scores)
    fold_correlations = np.array(fold_correlations)
    
    # Remove NaN values
    valid_scores = fold_scores[~np.isnan(fold_scores)]
    valid_correlations = fold_correlations[~np.isnan(fold_correlations)]
    
    metrics = {
        'mean_sharpe_like_score': np.mean(valid_scores) if len(valid_scores) > 0 else 0.0,
        'std_sharpe_like_score': np.std(valid_scores) if len(valid_scores) > 0 else 0.0,
        'mean_spearman_correlation': np.mean(valid_correlations) if len(valid_correlations) > 0 else 0.0,
        'std_spearman_correlation': np.std(valid_correlations) if len(valid_correlations) > 0 else 0.0,
        'n_valid_folds': len(valid_scores),
        'fold_scores': valid_scores.tolist()
    }
    
    return metrics


def calculate_stability_metrics(
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate prediction stability metrics.
    
    Args:
        y_pred: Predicted values, shape (n_samples, n_targets)
        
    Returns:
        Dictionary containing stability metrics
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    stability_metrics = {}
    
    # Temporal stability (across time)
    temporal_volatility = np.std(y_pred, axis=0)
    stability_metrics['mean_temporal_volatility'] = np.mean(temporal_volatility)
    stability_metrics['max_temporal_volatility'] = np.max(temporal_volatility)
    
    # Cross-target stability (across targets)
    if y_pred.shape[1] > 1:
        cross_target_volatility = np.std(y_pred, axis=1)
        stability_metrics['mean_cross_target_volatility'] = np.mean(cross_target_volatility)
        stability_metrics['max_cross_target_volatility'] = np.max(cross_target_volatility)
        
        # Correlation consistency
        target_correlations = np.corrcoef(y_pred.T)
        stability_metrics['mean_target_correlation'] = np.mean(target_correlations[np.triu_indices_from(target_correlations, k=1)])
        stability_metrics['std_target_correlation'] = np.std(target_correlations[np.triu_indices_from(target_correlations, k=1)])
    
    return stability_metrics


def calculate_economic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate economic performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing economic metrics
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    economic_metrics = {}
    
    # Calculate returns-based metrics
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    for i in range(y_true.shape[1]):
        true_returns = y_true[:, i]
        pred_returns = y_pred[:, i]
        
        # Information Ratio
        excess_returns = pred_returns - true_returns
        if np.std(excess_returns) > 0:
            info_ratio = np.mean(excess_returns) / np.std(excess_returns)
        else:
            info_ratio = 0.0
        
        # Hit Rate (directional accuracy)
        if len(true_returns) > 1:
            true_directions = np.sign(np.diff(true_returns))
            pred_directions = np.sign(np.diff(pred_returns))
            hit_rate = np.mean(true_directions == pred_directions)
        else:
            hit_rate = 0.0
        
        economic_metrics[f'information_ratio_target_{i}'] = info_ratio
        economic_metrics[f'hit_rate_target_{i}'] = hit_rate
    
    # Overall metrics
    economic_metrics['mean_information_ratio'] = np.mean([
        v for k, v in economic_metrics.items() if 'information_ratio' in k
    ])
    economic_metrics['mean_hit_rate'] = np.mean([
        v for k, v in economic_metrics.items() if 'hit_rate' in k
    ])
    
    return economic_metrics


class CompetitionEvaluator:
    """
    Comprehensive evaluator for the Mitsui Commodity Prediction Challenge.
    """
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_id: str = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            fold_id: Optional fold identifier
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Primary competition metric
        results['sharpe_like_score'] = calculate_sharpe_like_score(y_true, y_pred)
        
        # Correlation metrics
        if y_true.ndim > 1:
            correlations = []
            for i in range(y_true.shape[1]):
                corr = calculate_spearman_correlation(y_true[:, i], y_pred[:, i])
                correlations.append(corr)
            
            results['mean_spearman_correlation'] = np.mean(correlations)
            results['std_spearman_correlation'] = np.std(correlations)
            results['min_spearman_correlation'] = np.min(correlations)
            results['max_spearman_correlation'] = np.max(correlations)
        else:
            corr = calculate_spearman_correlation(y_true, y_pred)
            results['spearman_correlation'] = corr
        
        # Stability metrics
        stability_metrics = calculate_stability_metrics(y_pred)
        results.update(stability_metrics)
        
        # Economic metrics
        economic_metrics = calculate_economic_metrics(y_true, y_pred)
        results.update(economic_metrics)
        
        # Store evaluation history
        evaluation_record = {
            'fold_id': fold_id,
            'timestamp': pd.Timestamp.now(),
            'metrics': results.copy()
        }
        self.evaluation_history.append(evaluation_record)
        
        return results
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of all evaluations.
        
        Returns:
            DataFrame containing evaluation history
        """
        if not self.evaluation_history:
            return pd.DataFrame()
        
        summary_data = []
        for record in self.evaluation_history:
            row = {'fold_id': record['fold_id'], 'timestamp': record['timestamp']}
            row.update(record['metrics'])
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def validate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None
) -> Dict[str, any]:
    """
    Validate predictions and identify potential issues.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target_names: Optional target names
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Shape validation
    if y_true.shape != y_pred.shape:
        validation_results['errors'].append(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
        validation_results['is_valid'] = False
        return validation_results
    
    # NaN/Inf validation
    nan_true = np.isnan(y_true).sum()
    nan_pred = np.isnan(y_pred).sum()
    inf_true = np.isinf(y_true).sum()
    inf_pred = np.isinf(y_pred).sum()
    
    if nan_true > 0:
        validation_results['warnings'].append(f"Found {nan_true} NaN values in y_true")
    
    if nan_pred > 0:
        validation_results['warnings'].append(f"Found {nan_pred} NaN values in y_pred")
    
    if inf_true > 0:
        validation_results['warnings'].append(f"Found {inf_true} Inf values in y_true")
    
    if inf_pred > 0:
        validation_results['warnings'].append(f"Found {inf_pred} Inf values in y_pred")
    
    # Range validation
    if y_pred.ndim > 1:
        for i in range(y_pred.shape[1]):
            pred_range = np.max(y_pred[:, i]) - np.min(y_pred[:, i])
            if pred_range == 0:
                target_name = target_names[i] if target_names else f"target_{i}"
                validation_results['warnings'].append(
                    f"Zero variance in predictions for {target_name}"
                )
    
    return validation_results