"""
Advanced Feature Engineering for Mitsui Commodity Prediction Challenge.

Implements comprehensive feature engineering pipeline including:
- Technical indicators
- Cross-asset features  
- Economic factors
- Regime detection features
- Multi-horizon features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical analysis indicators for financial time series."""
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        bb_width = (upper_band - lower_band) / rolling_mean
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return {
            'bb_upper': upper_band,
            'bb_lower': lower_band, 
            'bb_middle': rolling_mean,
            'bb_width': bb_width,
            'bb_position': bb_position
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))


class CrossAssetFeatures:
    """Cross-asset relationship features."""
    
    @staticmethod
    def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int = 30) -> pd.Series:
        """Rolling correlation between two series."""
        return series1.rolling(window=window).corr(series2)
    
    @staticmethod
    def correlation_matrix_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Extract features from rolling correlation matrix."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        features = pd.DataFrame(index=df.index)
        
        # Calculate pairwise correlations
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                corr = df[col1].rolling(window=window).corr(df[col2])
                features[f'corr_{col1}_{col2}'] = corr
        
        return features
    
    @staticmethod
    def spread_features(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Calculate spread features between asset pairs."""
        features = pd.DataFrame(index=df.index)
        
        for col1, col2 in pairs:
            if col1 in df.columns and col2 in df.columns:
                # Price spread
                spread = df[col1] - df[col2]
                features[f'spread_{col1}_{col2}'] = spread
                
                # Ratio
                ratio = df[col1] / df[col2].replace(0, np.nan)
                features[f'ratio_{col1}_{col2}'] = ratio
                
                # Z-score of spread
                spread_zscore = (spread - spread.rolling(window=30).mean()) / spread.rolling(window=30).std()
                features[f'spread_zscore_{col1}_{col2}'] = spread_zscore
        
        return features
    
    @staticmethod
    def volatility_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30]) -> pd.DataFrame:
        """Calculate volatility features."""
        features = pd.DataFrame(index=df.index)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            returns = df[col].pct_change()
            
            for window in windows:
                # Rolling volatility
                vol = returns.rolling(window=window).std()
                features[f'{col}_vol_{window}'] = vol
                
                # Parkinson volatility (if we had OHLC data)
                # For now, use realized volatility
                features[f'{col}_realized_vol_{window}'] = np.sqrt(
                    (returns.rolling(window=window) ** 2).sum()
                )
        
        return features


class RegimeDetection:
    """Market regime detection features."""
    
    @staticmethod
    def volatility_regime(prices: pd.Series, window: int = 20, threshold: float = 1.5) -> pd.Series:
        """Detect volatility regimes."""
        returns = prices.pct_change()
        rolling_vol = returns.rolling(window=window).std()
        median_vol = rolling_vol.rolling(window=window*3).median()
        
        # High vol regime when current vol > threshold * median vol
        regime = (rolling_vol > threshold * median_vol).astype(int)
        return regime
    
    @staticmethod
    def trend_regime(prices: pd.Series, window: int = 20) -> pd.Series:
        """Detect trend regimes using moving averages."""
        ma_short = prices.rolling(window=window//2).mean()
        ma_long = prices.rolling(window=window).mean()
        
        # 1 = uptrend, 0 = downtrend
        regime = (ma_short > ma_long).astype(int)
        return regime
    
    @staticmethod
    def market_stress_indicator(df: pd.DataFrame) -> pd.Series:
        """Create market stress indicator from multiple assets."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate volatility for each asset
        volatilities = pd.DataFrame()
        for col in numeric_cols:
            returns = df[col].pct_change()
            vol = returns.rolling(window=20).std()
            volatilities[col] = vol
        
        # Average volatility across assets
        avg_vol = volatilities.mean(axis=1)
        
        # Normalize to create stress indicator (0-1)
        stress_indicator = (avg_vol - avg_vol.rolling(window=60).min()) / (
            avg_vol.rolling(window=60).max() - avg_vol.rolling(window=60).min()
        )
        
        return stress_indicator.fillna(0)


class EconomicFeatures:
    """Economic and fundamental features."""
    
    @staticmethod
    def momentum_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """Price momentum features."""
        features = pd.DataFrame(index=df.index)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            for window in windows:
                # Price momentum
                momentum = df[col] / df[col].shift(window) - 1
                features[f'{col}_momentum_{window}'] = momentum
                
                # ROC (Rate of Change)
                roc = df[col].pct_change(periods=window)
                features[f'{col}_roc_{window}'] = roc
        
        return features
    
    @staticmethod
    def seasonal_features(df: pd.DataFrame, date_col: str = 'date_id') -> pd.DataFrame:
        """Seasonal and calendar features."""
        features = pd.DataFrame(index=df.index)
        
        if date_col in df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                dates = pd.to_datetime(df[date_col], errors='coerce')
            else:
                dates = df[date_col]
            
            # Calendar features
            features['month'] = dates.dt.month
            features['quarter'] = dates.dt.quarter
            features['day_of_week'] = dates.dt.dayofweek
            features['day_of_month'] = dates.dt.day
            features['week_of_year'] = dates.dt.isocalendar().week
            
            # Cyclical encoding
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering pipeline for commodity prediction.
    """
    
    def __init__(
        self,
        technical_indicators: bool = True,
        cross_asset_features: bool = True,
        regime_features: bool = True,
        economic_features: bool = True,
        lag_features: List[int] = [1, 2, 3, 5, 7, 10, 15, 20, 30],
        rolling_windows: List[int] = [5, 10, 20, 30, 60],
        volatility_windows: List[int] = [5, 10, 20, 30],
        correlation_window: int = 30
    ):
        """
        Initialize feature engineering pipeline.
        
        Args:
            technical_indicators: Include technical indicator features
            cross_asset_features: Include cross-asset relationship features
            regime_features: Include regime detection features
            economic_features: Include economic/fundamental features
            lag_features: List of lag periods
            rolling_windows: List of rolling window sizes
            volatility_windows: Volatility calculation windows
            correlation_window: Window for correlation calculations
        """
        self.technical_indicators = technical_indicators
        self.cross_asset_features = cross_asset_features
        self.regime_features = regime_features
        self.economic_features = economic_features
        
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.volatility_windows = volatility_windows
        self.correlation_window = correlation_window
        
        # Initialize feature calculators
        self.tech_indicators = TechnicalIndicators()
        self.cross_asset = CrossAssetFeatures()
        self.regime_detector = RegimeDetection()
        self.economic = EconomicFeatures()
        
        # Feature tracking
        self.feature_names = []
        self.feature_groups = {}
    
    def create_features(self, df: pd.DataFrame, target_cols: List[str] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            df: Input data
            target_cols: Target column names (to exclude from feature creation)
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting advanced feature engineering...")
        
        # Ensure data is sorted by date
        if 'date_id' in df.columns:
            df = df.sort_values('date_id')
        
        # Get numeric columns (exclude targets)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_cols:
            numeric_cols = [col for col in numeric_cols if col not in target_cols]
        
        # Initialize features dataframe
        features = pd.DataFrame(index=df.index)
        
        # 1. Basic lag and rolling features
        if self.lag_features or self.rolling_windows:
            logger.info("Creating lag and rolling features...")
            lag_roll_features = self._create_lag_rolling_features(df[numeric_cols])
            features = pd.concat([features, lag_roll_features], axis=1)
        
        # 2. Technical indicators
        if self.technical_indicators:
            logger.info("Creating technical indicator features...")
            tech_features = self._create_technical_features(df[numeric_cols])
            features = pd.concat([features, tech_features], axis=1)
        
        # 3. Cross-asset features
        if self.cross_asset_features:
            logger.info("Creating cross-asset features...")
            cross_features = self._create_cross_asset_features(df[numeric_cols])
            features = pd.concat([features, cross_features], axis=1)
        
        # 4. Regime features
        if self.regime_features:
            logger.info("Creating regime detection features...")
            regime_features = self._create_regime_features(df[numeric_cols])
            features = pd.concat([features, regime_features], axis=1)
        
        # 5. Economic features
        if self.economic_features:
            logger.info("Creating economic features...")
            econ_features = self._create_economic_features(df)
            features = pd.concat([features, econ_features], axis=1)
        
        # Add original features (excluding targets)
        original_features = df[numeric_cols]
        features = pd.concat([original_features, features], axis=1)
        
        # Store feature information
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Feature engineering completed. Created {len(self.feature_names)} features.")
        
        return features
    
    def _create_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag and rolling window features."""
        features = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            # Lag features
            for lag in self.lag_features:
                features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Rolling statistics
            for window in self.rolling_windows:
                features[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                features[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
                features[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()
                features[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
                features[f'{col}_roll_median_{window}'] = df[col].rolling(window=window).median()
                
                # Rolling quantiles
                features[f'{col}_roll_q25_{window}'] = df[col].rolling(window=window).quantile(0.25)
                features[f'{col}_roll_q75_{window}'] = df[col].rolling(window=window).quantile(0.75)
                
                # Rolling skewness and kurtosis
                features[f'{col}_roll_skew_{window}'] = df[col].rolling(window=window).skew()
                features[f'{col}_roll_kurt_{window}'] = df[col].rolling(window=window).kurt()
        
        return features
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        features = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            prices = df[col]
            
            # RSI
            rsi = self.tech_indicators.rsi(prices)
            features[f'{col}_rsi'] = rsi
            
            # MACD
            macd_dict = self.tech_indicators.macd(prices)
            for name, series in macd_dict.items():
                features[f'{col}_macd_{name}'] = series
            
            # Bollinger Bands
            bb_dict = self.tech_indicators.bollinger_bands(prices)
            for name, series in bb_dict.items():
                features[f'{col}_bb_{name}'] = series
            
            # Williams %R
            # For simplicity, use price as high/low/close
            williams_r = self.tech_indicators.williams_r(prices, prices, prices)
            features[f'{col}_williams_r'] = williams_r
        
        return features
    
    def _create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset relationship features."""
        features = pd.DataFrame(index=df.index)
        
        # Volatility features
        vol_features = self.cross_asset.volatility_features(df, self.volatility_windows)
        features = pd.concat([features, vol_features], axis=1)
        
        # Correlation features (sample a subset to avoid too many features)
        cols = df.columns.tolist()
        if len(cols) > 10:
            # Sample pairs to avoid explosion of features
            import random
            random.seed(42)
            pairs = random.sample([(i, j) for i in cols for j in cols if i != j], 50)
        else:
            pairs = [(i, j) for i in cols for j in cols if i != j]
        
        for col1, col2 in pairs:
            corr = self.cross_asset.rolling_correlation(
                df[col1], df[col2], self.correlation_window
            )
            features[f'corr_{col1}_{col2}'] = corr
        
        return features
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime detection features."""
        features = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            # Volatility regime
            vol_regime = self.regime_detector.volatility_regime(df[col])
            features[f'{col}_vol_regime'] = vol_regime
            
            # Trend regime
            trend_regime = self.regime_detector.trend_regime(df[col])
            features[f'{col}_trend_regime'] = trend_regime
        
        # Market stress indicator
        stress_indicator = self.regime_detector.market_stress_indicator(df)
        features['market_stress'] = stress_indicator
        
        return features
    
    def _create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create economic and fundamental features."""
        features = pd.DataFrame(index=df.index)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Momentum features
        momentum_features = self.economic.momentum_features(df[numeric_cols])
        features = pd.concat([features, momentum_features], axis=1)
        
        # Seasonal features
        seasonal_features = self.economic.seasonal_features(df)
        features = pd.concat([features, seasonal_features], axis=1)
        
        return features
    
    def get_feature_importance_analysis(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        Analyze feature importance using various methods.
        
        Args:
            features: Feature matrix
            target: Target variable
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.feature_selection import mutual_info_regression, f_regression
        from sklearn.ensemble import RandomForestRegressor
        
        # Remove rows with NaN values
        valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
        X_clean = features[valid_idx]
        y_clean = target[valid_idx]
        
        if len(X_clean) == 0:
            logger.warning("No valid samples for feature importance analysis")
            return pd.DataFrame()
        
        importance_scores = pd.DataFrame(index=features.columns)
        
        # Mutual information
        try:
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            importance_scores['mutual_info'] = mi_scores
        except Exception as e:
            logger.warning(f"Could not calculate mutual information: {e}")
        
        # F-statistic
        try:
            f_scores, _ = f_regression(X_clean, y_clean)
            importance_scores['f_statistic'] = f_scores
        except Exception as e:
            logger.warning(f"Could not calculate F-statistics: {e}")
        
        # Random Forest importance
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_clean, y_clean)
            importance_scores['rf_importance'] = rf.feature_importances_
        except Exception as e:
            logger.warning(f"Could not calculate RF importance: {e}")
        
        # Correlation with target
        try:
            correlations = X_clean.corrwith(y_clean).abs()
            importance_scores['correlation'] = correlations
        except Exception as e:
            logger.warning(f"Could not calculate correlations: {e}")
        
        return importance_scores.fillna(0)


def make_features(
    df: pd.DataFrame, 
    target_cols: List[str] = None,
    feature_config: Dict = None
) -> pd.DataFrame:
    """
    Main feature engineering function (backward compatibility).
    
    Args:
        df: Input dataframe
        target_cols: Target column names to exclude
        feature_config: Configuration for feature engineering
        
    Returns:
        DataFrame with engineered features
    """
    if feature_config is None:
        feature_config = {}
    
    engineer = AdvancedFeatureEngineer(**feature_config)
    return engineer.create_features(df, target_cols)