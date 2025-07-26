"""
AI-Driven Experiment Management System for Mitsui Competition.

Automatically manages experiments, optimizes hyperparameters, and generates insights
to accelerate model development and identify winning strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import logging
from pathlib import Path
from datetime import datetime
import pickle
from dataclasses import dataclass, asdict
import uuid

# For Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    optuna = None
    logging.warning("Optuna not available. Install with: pip install optuna")

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    name: str
    model_type: str
    feature_config: Dict[str, Any]
    model_params: Dict[str, Any]
    cv_strategy: str
    evaluation_metrics: List[str]
    resource_limits: Dict[str, Any]
    priority: int = 1
    notes: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ExperimentResult:
    """Results from a completed experiment."""
    experiment_id: str
    config: ExperimentConfig
    performance_metrics: Dict[str, float]
    model_artifacts: Dict[str, Any]
    training_time: float
    memory_usage: float
    status: str  # 'completed', 'failed', 'timeout'
    error_message: str = ""
    insights: List[str] = None
    completed_at: str = ""
    
    def __post_init__(self):
        if not self.completed_at:
            self.completed_at = datetime.now().isoformat()
        if self.insights is None:
            self.insights = []


class PerformanceDatabase:
    """Database for storing and querying experiment results."""
    
    def __init__(self, db_path: str = "experiments/performance_db.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        self.load_database()
    
    def save_database(self):
        """Save database to disk."""
        data = [asdict(result) for result in self.results]
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_database(self):
        """Load database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                
                self.results = []
                for item in data:
                    config_dict = item['config']
                    config = ExperimentConfig(**config_dict)
                    
                    result = ExperimentResult(
                        experiment_id=item['experiment_id'],
                        config=config,
                        performance_metrics=item['performance_metrics'],
                        model_artifacts=item['model_artifacts'],
                        training_time=item['training_time'],
                        memory_usage=item['memory_usage'],
                        status=item['status'],
                        error_message=item.get('error_message', ''),
                        insights=item.get('insights', []),
                        completed_at=item.get('completed_at', '')
                    )
                    self.results.append(result)
                    
                logger.info(f"Loaded {len(self.results)} experiment results from database")
                
            except Exception as e:
                logger.warning(f"Could not load database: {e}")
                self.results = []
    
    def add_result(self, result: ExperimentResult):
        """Add experiment result to database."""
        self.results.append(result)
        self.save_database()
        logger.info(f"Added experiment result: {result.experiment_id}")
    
    def get_best_results(self, metric: str = 'sharpe_like_score', top_k: int = 10) -> List[ExperimentResult]:
        """Get top performing experiments by metric."""
        completed_results = [r for r in self.results if r.status == 'completed']
        
        if not completed_results:
            return []
        
        # Sort by metric (descending)
        sorted_results = sorted(
            completed_results,
            key=lambda x: x.performance_metrics.get(metric, -np.inf),
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def get_experiment_trends(self, metric: str = 'sharpe_like_score') -> pd.DataFrame:
        """Analyze performance trends over time."""
        data = []
        for result in self.results:
            if result.status == 'completed' and metric in result.performance_metrics:
                data.append({
                    'experiment_id': result.experiment_id,
                    'completed_at': result.completed_at,
                    'model_type': result.config.model_type,
                    'metric_value': result.performance_metrics[metric],
                    'training_time': result.training_time,
                    'memory_usage': result.memory_usage
                })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['completed_at'] = pd.to_datetime(df['completed_at'])
        return df.sort_values('completed_at')
    
    def get_model_performance_comparison(self) -> pd.DataFrame:
        """Compare performance across different model types."""
        data = []
        for result in self.results:
            if result.status == 'completed':
                row = {
                    'model_type': result.config.model_type,
                    'experiment_id': result.experiment_id,
                    **result.performance_metrics
                }
                data.append(row)
        
        return pd.DataFrame(data) if data else pd.DataFrame()


class AIInsightEngine:
    """AI system for generating insights from experiment results."""
    
    def __init__(self, performance_db: PerformanceDatabase):
        self.performance_db = performance_db
    
    def analyze_feature_importance_trends(self) -> Dict[str, Any]:
        """Analyze how feature importance evolves across experiments."""
        trends = {}
        
        # Analyze completed experiments with feature importance data
        for result in self.performance_db.results:
            if (result.status == 'completed' and 
                'feature_importance' in result.model_artifacts):
                
                importance_data = result.model_artifacts['feature_importance']
                model_type = result.config.model_type
                
                if model_type not in trends:
                    trends[model_type] = []
                
                trends[model_type].append({
                    'experiment_id': result.experiment_id,
                    'performance': result.performance_metrics.get('sharpe_like_score', 0),
                    'importance': importance_data
                })
        
        # Generate insights
        insights = {}
        for model_type, experiments in trends.items():
            if len(experiments) >= 3:  # Need multiple experiments for trends
                insights[model_type] = self._identify_important_features(experiments)
        
        return insights
    
    def _identify_important_features(self, experiments: List[Dict]) -> Dict[str, Any]:
        """Identify consistently important features across experiments."""
        # Sort by performance
        experiments = sorted(experiments, key=lambda x: x['performance'], reverse=True)
        
        # Analyze top-performing experiments
        top_experiments = experiments[:min(5, len(experiments))]
        
        feature_scores = {}
        for exp in top_experiments:
            if isinstance(exp['importance'], dict):
                for feature, score in exp['importance'].items():
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(score)
        
        # Calculate average and consistency
        consistent_features = {}
        for feature, scores in feature_scores.items():
            if len(scores) >= 3:  # Feature appears in multiple top experiments
                consistent_features[feature] = {
                    'avg_importance': np.mean(scores),
                    'consistency': 1.0 - (np.std(scores) / (np.mean(scores) + 1e-8)),
                    'frequency': len(scores) / len(top_experiments)
                }
        
        return consistent_features
    
    def detect_performance_patterns(self) -> List[str]:
        """Detect patterns in experiment performance."""
        insights = []
        
        df = self.performance_db.get_experiment_trends()
        if df.empty:
            return ["No experiments completed yet."]
        
        # Trend analysis
        if len(df) >= 5:
            recent_performance = df.tail(5)['metric_value'].mean()
            early_performance = df.head(5)['metric_value'].mean()
            
            if recent_performance > early_performance * 1.1:
                insights.append("📈 Performance improving over time - good research direction!")
            elif recent_performance < early_performance * 0.9:
                insights.append("📉 Recent experiments underperforming - consider revisiting successful approaches")
        
        # Model type analysis
        model_comparison = self.performance_db.get_model_performance_comparison()
        if not model_comparison.empty and 'sharpe_like_score' in model_comparison.columns:
            best_models = model_comparison.groupby('model_type')['sharpe_like_score'].mean().sort_values(ascending=False)
            
            if len(best_models) > 1:
                best_model = best_models.index[0]
                best_score = best_models.iloc[0]
                insights.append(f"🏆 Best performing model type: {best_model} (avg score: {best_score:.4f})")
                
                if len(best_models) > 2:
                    underperforming = best_models.index[-1]
                    insights.append(f"⚠️  Consider deprioritizing: {underperforming}")
        
        # Resource efficiency analysis
        if 'training_time' in df.columns and 'metric_value' in df.columns:
            # Find experiments with good performance/time ratio
            df['efficiency'] = df['metric_value'] / (df['training_time'] + 1)
            top_efficient = df.nlargest(3, 'efficiency')
            
            if not top_efficient.empty:
                avg_time = top_efficient['training_time'].mean()
                insights.append(f"⚡ Most efficient experiments average {avg_time:.1f}s training time")
        
        return insights
    
    def suggest_next_experiments(self, current_best_score: float) -> List[Dict[str, Any]]:
        """Suggest promising next experiments based on historical data."""
        suggestions = []
        
        # Analyze what works
        best_results = self.performance_db.get_best_results(top_k=5)
        if not best_results:
            return [{"suggestion": "Run baseline experiments first", "priority": 1}]
        
        # Pattern analysis
        successful_configs = []
        for result in best_results:
            config = result.config
            successful_configs.append({
                'model_type': config.model_type,
                'feature_config': config.feature_config,
                'model_params': config.model_params,
                'performance': result.performance_metrics.get('sharpe_like_score', 0)
            })
        
        # Suggest variations of successful approaches
        for i, config in enumerate(successful_configs[:3]):
            suggestion = {
                'suggestion': f"Variation of successful {config['model_type']} approach",
                'model_type': config['model_type'],
                'priority': i + 1,
                'reasoning': f"Based on experiment with score {config['performance']:.4f}",
                'modifications': self._suggest_modifications(config)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_modifications(self, config: Dict[str, Any]) -> List[str]:
        """Suggest modifications to a successful configuration."""
        modifications = []
        
        model_type = config['model_type']
        
        if model_type == 'lightgbm':
            modifications.extend([
                "Increase n_estimators by 50%",
                "Reduce learning_rate by 25%",
                "Adjust feature_fraction ±0.1",
                "Try different regularization (alpha/lambda)"
            ])
        elif 'neural' in model_type.lower():
            modifications.extend([
                "Add dropout layers",
                "Increase/decrease hidden size",
                "Try different activation functions",
                "Adjust learning rate schedule"
            ])
        
        modifications.extend([
            "Add more advanced features",
            "Try different CV strategy",
            "Adjust ensemble weights"
        ])
        
        return modifications[:3]  # Return top 3 suggestions


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, study_name: str = "mitsui_optimization"):
        if optuna is None:
            raise ImportError("Optuna required for Bayesian optimization")
        
        self.study_name = study_name
        self.study = None
    
    def create_study(self, direction: str = 'maximize', metric_name: str = 'sharpe_like_score'):
        """Create or load optimization study."""
        self.study = optuna.create_study(
            study_name=f"{self.study_name}_{metric_name}",
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True
        )
        logger.info(f"Created optimization study: {self.study.study_name}")
    
    def suggest_hyperparameters(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for given model type."""
        params = {}
        
        if model_type == 'lightgbm':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
            })
        
        elif model_type == 'xgboost':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            })
        
        elif model_type == 'neural_network':
            params.update({
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
                'num_layers': trial.suggest_int('num_layers', 1, 5),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'gelu'])
            })
        
        return params
    
    def optimize(self, objective_func, n_trials: int = 100, timeout: int = 3600):
        """Run Bayesian optimization."""
        if self.study is None:
            self.create_study()
        
        logger.info(f"Starting optimization with {n_trials} trials, timeout {timeout}s")
        
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,)
        )
        
        logger.info("Optimization completed")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.study.best_params}")
        
        return self.study.best_params, self.study.best_value


class IntelligentExperimentRunner:
    """Main experiment management system."""
    
    def __init__(self, db_path: str = "experiments/performance_db.json"):
        self.performance_db = PerformanceDatabase(db_path)
        self.insight_engine = AIInsightEngine(self.performance_db)
        self.bayesian_optimizer = None
        
        # Initialize Bayesian optimizer if available
        if optuna is not None:
            try:
                self.bayesian_optimizer = BayesianOptimizer()
            except Exception as e:
                logger.warning(f"Could not initialize Bayesian optimizer: {e}")
    
    def run_experiment(self, config: ExperimentConfig, objective_func) -> ExperimentResult:
        """Run a single experiment."""
        logger.info(f"Running experiment: {config.name} ({config.experiment_id})")
        
        start_time = datetime.now()
        
        try:
            # Run the actual experiment
            result = objective_func(config)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            experiment_result = ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                performance_metrics=result.get('metrics', {}),
                model_artifacts=result.get('artifacts', {}),
                training_time=training_time,
                memory_usage=result.get('memory_usage', 0),
                status='completed'
            )
            
            # Add to database
            self.performance_db.add_result(experiment_result)
            
            logger.info(f"Experiment completed: {config.experiment_id}")
            logger.info(f"Performance: {experiment_result.performance_metrics}")
            
            return experiment_result
            
        except Exception as e:
            # Handle failed experiment
            training_time = (datetime.now() - start_time).total_seconds()
            
            experiment_result = ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                performance_metrics={},
                model_artifacts={},
                training_time=training_time,
                memory_usage=0,
                status='failed',
                error_message=str(e)
            )
            
            self.performance_db.add_result(experiment_result)
            
            logger.error(f"Experiment failed: {config.experiment_id} - {e}")
            return experiment_result
    
    def generate_experiment_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        # Get performance trends
        trends_df = self.performance_db.get_experiment_trends()
        model_comparison_df = self.performance_db.get_model_performance_comparison()
        
        # Get AI insights
        insights = self.insight_engine.detect_performance_patterns()
        suggestions = self.insight_engine.suggest_next_experiments(
            current_best_score=self._get_current_best_score()
        )
        
        # Best results
        best_results = self.performance_db.get_best_results(top_k=5)
        
        report = {
            'summary': {
                'total_experiments': len(self.performance_db.results),
                'completed_experiments': len([r for r in self.performance_db.results if r.status == 'completed']),
                'failed_experiments': len([r for r in self.performance_db.results if r.status == 'failed']),
                'best_score': self._get_current_best_score(),
                'generated_at': datetime.now().isoformat()
            },
            'performance_trends': trends_df.to_dict('records') if not trends_df.empty else [],
            'model_comparison': model_comparison_df.to_dict('records') if not model_comparison_df.empty else [],
            'ai_insights': insights,
            'suggested_experiments': suggestions,
            'best_experiments': [
                {
                    'experiment_id': r.experiment_id,
                    'model_type': r.config.model_type,
                    'performance': r.performance_metrics,
                    'config_summary': {
                        'name': r.config.name,
                        'notes': r.config.notes
                    }
                }
                for r in best_results
            ]
        }
        
        return report
    
    def _get_current_best_score(self) -> float:
        """Get current best Sharpe-like score."""
        best_results = self.performance_db.get_best_results(top_k=1)
        if best_results:
            return best_results[0].performance_metrics.get('sharpe_like_score', 0.0)
        return 0.0
    
    def save_experiment_report(self, filepath: str = None):
        """Save experiment report to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"experiments/experiment_report_{timestamp}.json"
        
        report = self.generate_experiment_report()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Experiment report saved to: {filepath}")
        return filepath


def create_experiment_config(
    name: str,
    model_type: str,
    feature_config: Dict[str, Any] = None,
    model_params: Dict[str, Any] = None,
    notes: str = ""
) -> ExperimentConfig:
    """Helper function to create experiment configuration."""
    
    experiment_id = str(uuid.uuid4())[:8]
    
    return ExperimentConfig(
        experiment_id=experiment_id,
        name=name,
        model_type=model_type,
        feature_config=feature_config or {},
        model_params=model_params or {},
        cv_strategy='time_series',
        evaluation_metrics=['sharpe_like_score', 'spearman_correlation', 'stability_score'],
        resource_limits={'max_training_time': 3600, 'max_memory_gb': 16},
        notes=notes
    )