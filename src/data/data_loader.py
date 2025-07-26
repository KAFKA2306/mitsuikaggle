"""
Data loading and preprocessing for Mitsui Commodity Prediction Challenge.

Handles loading of competition data, basic preprocessing, and data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import warnings
import logging

logger = logging.getLogger(__name__)


class MitsuiDataLoader:
    """
    Data loader for Mitsui Commodity Prediction Challenge.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "input"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing competition data
        """
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
        
        # Data storage
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.target_pairs = None
        self.lagged_test_labels = {}
        
        # Metadata
        self.feature_columns = None
        self.target_columns = None
        self.data_info = {}
    
    def _validate_data_dir(self):
        """Validate that data directory contains required files."""
        required_files = [
            'train.csv',
            'train_labels.csv', 
            'test.csv',
            'target_pairs.csv'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {self.data_dir}: {missing_files}"
            )
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data.
        
        Returns:
            Training data DataFrame
        """
        logger.info("Loading training data...")
        
        self.train_data = pd.read_csv(self.data_dir / 'train.csv')
        
        # Basic validation
        logger.info(f"Loaded training data: {self.train_data.shape}")
        logger.info(f"Date range: {self.train_data['date_id'].min()} to {self.train_data['date_id'].max()}")
        
        # Store feature columns (exclude date_id)
        self.feature_columns = [col for col in self.train_data.columns if col != 'date_id']
        
        # Data quality info
        self.data_info['train_shape'] = self.train_data.shape
        self.data_info['train_missing'] = self.train_data.isnull().sum().sum()
        self.data_info['train_memory_mb'] = self.train_data.memory_usage(deep=True).sum() / 1024**2
        
        return self.train_data
    
    def load_train_labels(self) -> pd.DataFrame:
        """
        Load training labels.
        
        Returns:
            Training labels DataFrame
        """
        logger.info("Loading training labels...")
        
        self.train_labels = pd.read_csv(self.data_dir / 'train_labels.csv')
        
        # Basic validation
        logger.info(f"Loaded training labels: {self.train_labels.shape}")
        
        # Store target columns (exclude date_id)
        self.target_columns = [col for col in self.train_labels.columns if col != 'date_id']
        
        # Data quality info
        self.data_info['labels_shape'] = self.train_labels.shape
        self.data_info['labels_missing'] = self.train_labels.isnull().sum().sum()
        self.data_info['n_targets'] = len(self.target_columns)
        
        return self.train_labels
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data.
        
        Returns:
            Test data DataFrame
        """
        logger.info("Loading test data...")
        
        self.test_data = pd.read_csv(self.data_dir / 'test.csv')
        
        # Basic validation
        logger.info(f"Loaded test data: {self.test_data.shape}")
        
        self.data_info['test_shape'] = self.test_data.shape
        self.data_info['test_missing'] = self.test_data.isnull().sum().sum()
        
        return self.test_data
    
    def load_target_pairs(self) -> pd.DataFrame:
        """
        Load target pairs definition.
        
        Returns:
            Target pairs DataFrame
        """
        logger.info("Loading target pairs...")
        
        self.target_pairs = pd.read_csv(self.data_dir / 'target_pairs.csv')
        
        logger.info(f"Loaded target pairs: {self.target_pairs.shape}")
        logger.info(f"Unique lags: {sorted(self.target_pairs['lag'].unique())}")
        logger.info(f"Unique pairs: {self.target_pairs['pair'].nunique()}")
        
        return self.target_pairs
    
    def load_lagged_test_labels(self) -> Dict[int, pd.DataFrame]:
        """
        Load lagged test labels.
        
        Returns:
            Dictionary mapping lag to DataFrame
        """
        logger.info("Loading lagged test labels...")
        
        lagged_dir = self.data_dir / 'lagged_test_labels'
        if not lagged_dir.exists():
            logger.warning("Lagged test labels directory not found")
            return {}
        
        for lag in [1, 2, 3, 4]:
            file_path = lagged_dir / f'test_labels_lag_{lag}.csv'
            if file_path.exists():
                self.lagged_test_labels[lag] = pd.read_csv(file_path)
                logger.info(f"Loaded lag {lag} test labels: {self.lagged_test_labels[lag].shape}")
        
        return self.lagged_test_labels
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all competition data.
        
        Returns:
            Dictionary containing all loaded data
        """
        logger.info("Loading all competition data...")
        
        data = {
            'train': self.load_train_data(),
            'train_labels': self.load_train_labels(),
            'test': self.load_test_data(),
            'target_pairs': self.load_target_pairs(),
            'lagged_test_labels': self.load_lagged_test_labels()
        }
        
        # Validate consistency
        self._validate_data_consistency()
        
        logger.info("Data loading completed successfully")
        self._print_data_summary()
        
        return data
    
    def _validate_data_consistency(self):
        """Validate consistency across different data files."""
        issues = []
        
        # Check date_id consistency between train data and labels
        if self.train_data is not None and self.train_labels is not None:
            train_dates = set(self.train_data['date_id'])
            label_dates = set(self.train_labels['date_id'])
            
            if train_dates != label_dates:
                missing_in_labels = train_dates - label_dates
                missing_in_train = label_dates - train_dates
                
                if missing_in_labels:
                    issues.append(f"Date IDs in train but not in labels: {len(missing_in_labels)}")
                if missing_in_train:
                    issues.append(f"Date IDs in labels but not in train: {len(missing_in_train)}")
        
        # Check feature consistency between train and test
        if self.train_data is not None and self.test_data is not None:
            train_features = set(self.train_data.columns) - {'date_id'}
            test_features = set(self.test_data.columns) - {'date_id'}
            
            if train_features != test_features:
                missing_in_test = train_features - test_features
                extra_in_test = test_features - train_features
                
                if missing_in_test:
                    issues.append(f"Features in train but not in test: {missing_in_test}")
                if extra_in_test:
                    issues.append(f"Features in test but not in train: {extra_in_test}")
        
        # Check target pairs consistency
        if self.train_labels is not None and self.target_pairs is not None:
            expected_targets = set(self.target_pairs['target'])
            actual_targets = set(self.target_columns)
            
            if expected_targets != actual_targets:
                missing_targets = expected_targets - actual_targets
                extra_targets = actual_targets - expected_targets
                
                if missing_targets:
                    issues.append(f"Targets in pairs but not in labels: {missing_targets}")
                if extra_targets:
                    issues.append(f"Targets in labels but not in pairs: {extra_targets}")
        
        if issues:
            logger.warning("Data consistency issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Data consistency validation passed")
    
    def _print_data_summary(self):
        """Print summary of loaded data."""
        logger.info("\n" + "="*50)
        logger.info("DATA LOADING SUMMARY")
        logger.info("="*50)
        
        if self.train_data is not None:
            logger.info(f"Training Data: {self.train_data.shape}")
            logger.info(f"  - Features: {len(self.feature_columns)}")
            logger.info(f"  - Missing values: {self.data_info.get('train_missing', 0):,}")
            logger.info(f"  - Memory usage: {self.data_info.get('train_memory_mb', 0):.1f} MB")
        
        if self.train_labels is not None:
            logger.info(f"Training Labels: {self.train_labels.shape}")
            logger.info(f"  - Targets: {len(self.target_columns)}")
            logger.info(f"  - Missing values: {self.data_info.get('labels_missing', 0):,}")
        
        if self.test_data is not None:
            logger.info(f"Test Data: {self.test_data.shape}")
            logger.info(f"  - Missing values: {self.data_info.get('test_missing', 0):,}")
        
        if self.target_pairs is not None:
            logger.info(f"Target Pairs: {self.target_pairs.shape}")
            lag_counts = self.target_pairs['lag'].value_counts().sort_index()
            for lag, count in lag_counts.items():
                logger.info(f"  - Lag {lag}: {count} targets")
        
        if self.lagged_test_labels:
            logger.info(f"Lagged Test Labels: {len(self.lagged_test_labels)} lag files")
        
        logger.info("="*50)
    
    def get_feature_info(self) -> pd.DataFrame:
        """
        Get information about features.
        
        Returns:
            DataFrame with feature information
        """
        if self.train_data is None:
            raise ValueError("Training data not loaded")
        
        feature_info = []
        
        for col in self.feature_columns:
            info = {
                'feature': col,
                'dtype': str(self.train_data[col].dtype),
                'missing_count': self.train_data[col].isnull().sum(),
                'missing_pct': self.train_data[col].isnull().mean() * 100,
                'unique_values': self.train_data[col].nunique(),
                'min_value': self.train_data[col].min() if pd.api.types.is_numeric_dtype(self.train_data[col]) else None,
                'max_value': self.train_data[col].max() if pd.api.types.is_numeric_dtype(self.train_data[col]) else None,
                'mean_value': self.train_data[col].mean() if pd.api.types.is_numeric_dtype(self.train_data[col]) else None,
                'std_value': self.train_data[col].std() if pd.api.types.is_numeric_dtype(self.train_data[col]) else None
            }
            feature_info.append(info)
        
        return pd.DataFrame(feature_info)
    
    def get_target_info(self) -> pd.DataFrame:
        """
        Get information about targets.
        
        Returns:
            DataFrame with target information
        """
        if self.train_labels is None:
            raise ValueError("Training labels not loaded")
        
        if self.target_pairs is None:
            raise ValueError("Target pairs not loaded")
        
        target_info = []
        
        for col in self.target_columns:
            # Get target pair information
            pair_info = self.target_pairs[self.target_pairs['target'] == col].iloc[0]
            
            info = {
                'target': col,
                'lag': pair_info['lag'],
                'pair': pair_info['pair'],
                'missing_count': self.train_labels[col].isnull().sum(),
                'missing_pct': self.train_labels[col].isnull().mean() * 100,
                'min_value': self.train_labels[col].min(),
                'max_value': self.train_labels[col].max(),
                'mean_value': self.train_labels[col].mean(),
                'std_value': self.train_labels[col].std(),
                'skewness': self.train_labels[col].skew(),
                'kurtosis': self.train_labels[col].kurtosis()
            }
            target_info.append(info)
        
        return pd.DataFrame(target_info)
    
    def get_missing_data_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze missing data patterns.
        
        Returns:
            Dictionary containing missing data analysis
        """
        analysis = {}
        
        if self.train_data is not None:
            train_missing = self.train_data.isnull().sum()
            train_missing = train_missing[train_missing > 0].sort_values(ascending=False)
            analysis['train_missing'] = train_missing.to_frame('missing_count')
            analysis['train_missing']['missing_pct'] = (train_missing / len(self.train_data)) * 100
        
        if self.train_labels is not None:
            labels_missing = self.train_labels.isnull().sum()
            labels_missing = labels_missing[labels_missing > 0].sort_values(ascending=False)
            analysis['labels_missing'] = labels_missing.to_frame('missing_count')
            analysis['labels_missing']['missing_pct'] = (labels_missing / len(self.train_labels)) * 100
        
        if self.test_data is not None:
            test_missing = self.test_data.isnull().sum()
            test_missing = test_missing[test_missing > 0].sort_values(ascending=False)
            analysis['test_missing'] = test_missing.to_frame('missing_count')
            analysis['test_missing']['missing_pct'] = (test_missing / len(self.test_data)) * 100
        
        return analysis
    
    def prepare_model_data(
        self, 
        drop_missing_targets: bool = True,
        fill_missing_features: str = 'median'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            drop_missing_targets: Whether to remove samples with missing targets
            fill_missing_features: Strategy for filling missing features ('median', 'mean', 'zero')
            
        Returns:
            Tuple of (features, targets)
        """
        if self.train_data is None or self.train_labels is None:
            raise ValueError("Training data and labels must be loaded first")
        
        # Merge data on date_id
        merged_data = self.train_data.merge(
            self.train_labels, 
            on='date_id', 
            how='inner'
        )
        
        logger.info(f"Merged training data shape: {merged_data.shape}")
        
        # Prepare features
        feature_data = merged_data[self.feature_columns].copy()
        
        # Handle missing features
        if fill_missing_features == 'median':
            feature_data = feature_data.fillna(feature_data.median())
        elif fill_missing_features == 'mean':
            feature_data = feature_data.fillna(feature_data.mean())
        elif fill_missing_features == 'zero':
            feature_data = feature_data.fillna(0)
        else:
            raise ValueError(f"Unknown fill strategy: {fill_missing_features}")
        
        # Prepare targets
        target_data = merged_data[self.target_columns].copy()
        
        # Handle missing targets
        if drop_missing_targets:
            # Keep only rows where all targets are non-null
            valid_rows = target_data.notnull().all(axis=1)
            feature_data = feature_data[valid_rows]
            target_data = target_data[valid_rows]
            
            logger.info(f"After dropping missing targets: {feature_data.shape}")
        
        # Convert to numpy arrays
        X = feature_data.values.astype(np.float32)
        y = target_data.values.astype(np.float32)
        
        logger.info(f"Final data shapes - X: {X.shape}, y: {y.shape}")
        
        return X, y