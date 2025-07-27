#!/usr/bin/env python3
"""
Ultra Prediction Accuracy Experimental Pipeline
Systematic verification of improvement strategies for 2.0+ Sharpe Score

Test Matrix:
1. Baseline: Current 1.1912 Sharpe model
2. Phase 1: Advanced features + Ensemble (Target: 1.35 Sharpe)
3. Phase 2: Next-gen architecture (Target: 1.6 Sharpe)
4. Phase 3: Ultimate system preview (Target: 1.8+ Sharpe)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
warnings.filterwarnings('ignore')

# ========================= EXPERIMENT FRAMEWORK =========================

class ExperimentTracker:
    """Track all experiments with comprehensive metrics."""
    
    def __init__(self):
        self.experiments = []
        self.baseline_sharpe = 1.1912  # Current best
        
    def log_experiment(self, name: str, sharpe: float, metrics: dict, duration: float):
        """Log experiment results."""
        improvement = ((sharpe - self.baseline_sharpe) / self.baseline_sharpe) * 100
        
        experiment = {
            'name': name,
            'sharpe_score': sharpe,
            'improvement_pct': improvement,
            'duration_minutes': duration,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.experiments.append(experiment)
        
        print(f"🧪 {name}")
        print(f"   Sharpe: {sharpe:.4f} ({improvement:+.1f}%)")
        print(f"   Duration: {duration:.2f}min")
        print("   " + "="*50)
        
    def get_summary(self) -> pd.DataFrame:
        """Get experiment summary."""
        return pd.DataFrame(self.experiments)

# ========================= ADVANCED FEATURE ENGINEERING =========================

class AdvancedFeatureEngineer:
    """Next-generation feature engineering pipeline."""
    
    def __init__(self):
        self.scalers = {}
        
    def create_volatility_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect volatility regimes using rolling statistics."""
        features = pd.DataFrame(index=data.index)
        
        # Rolling volatility with multiple windows
        for window in [5, 10, 20]:
            vol_col = f'volatility_{window}d'
            features[vol_col] = data.select_dtypes(include=[np.number]).rolling(window).std().mean(axis=1)
            
            # Volatility regime (high/low based on percentiles)
            features[f'vol_regime_{window}d'] = pd.qcut(
                features[vol_col].fillna(method='ffill'), 
                q=3, labels=[0, 1, 2]
            ).astype(float)
            
        return features
    
    def create_cross_asset_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cross-asset momentum and divergence signals."""
        features = pd.DataFrame(index=data.index)
        
        # Identify asset groups by prefix
        lme_cols = [col for col in data.columns if 'LME_' in col]
        jpx_cols = [col for col in data.columns if 'JPX_' in col and 'Close' in col]
        us_cols = [col for col in data.columns if 'US_Stock_' in col and '_close' in col]
        fx_cols = [col for col in data.columns if 'FX_' in col]
        
        # Calculate group momentum
        for name, cols in [('LME', lme_cols), ('JPX', jpx_cols), ('US', us_cols), ('FX', fx_cols)]:
            if len(cols) > 0:
                group_data = data[cols].fillna(method='ffill')
                
                # 1, 3, 5 day momentum
                for period in [1, 3, 5]:
                    momentum = group_data.pct_change(period).mean(axis=1)
                    features[f'{name}_momentum_{period}d'] = momentum
                    
                # Cross-group correlations (momentum divergence)
                if name != 'FX':  # FX as reference
                    fx_momentum = data[fx_cols[:5]].fillna(method='ffill').pct_change(3).mean(axis=1)
                    group_momentum = group_data.pct_change(3).mean(axis=1)
                    
                    # Rolling correlation
                    features[f'{name}_FX_corr_20d'] = group_momentum.rolling(20).corr(fx_momentum)
                    
                    # Momentum divergence signal
                    features[f'{name}_FX_divergence'] = (group_momentum - fx_momentum).rolling(5).mean()
        
        return features
    
    def create_temporal_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Time-based pattern features."""
        features = pd.DataFrame(index=data.index)
        
        # Assuming data has date information
        if 'date_id' in data.columns:
            # Convert date_id to datetime-like features
            features['day_of_week'] = (data['date_id'] % 7)  # Proxy for day of week
            features['week_of_month'] = ((data['date_id'] % 30) // 7)  # Proxy for week
            features['month_proxy'] = ((data['date_id'] % 365) // 30)  # Proxy for month
            
            # Cyclical encoding
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month_proxy'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month_proxy'] / 12)
        
        # Lagged features (trend detection)
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:20]  # First 20 numeric
        for col in numeric_cols:
            if col != 'date_id':
                # Price acceleration (second derivative)
                price_diff = data[col].diff()
                features[f'{col}_acceleration'] = price_diff.diff()
                
                # Trend strength
                features[f'{col}_trend_5d'] = data[col].rolling(5).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 5 else 0
                )
        
        return features
    
    def create_correlation_cluster_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dynamic correlation clustering features."""
        features = pd.DataFrame(index=data.index)
        
        # Rolling correlation matrices for major asset groups
        major_assets = data.select_dtypes(include=[np.number]).columns[:50]  # Top 50 features
        
        # Calculate rolling correlation centrality measures
        window = 20
        for i in range(window, len(data)):
            window_data = data[major_assets].iloc[i-window:i]
            
            if not window_data.empty:
                corr_matrix = window_data.corr().fillna(0)
                
                # Correlation centrality (average correlation with others)
                centrality = corr_matrix.abs().mean()
                
                # Store top 10 centrality scores
                for j, asset in enumerate(centrality.nlargest(10).index):
                    features.loc[data.index[i], f'corr_centrality_{j}'] = centrality[asset]
        
        return features.fillna(method='ffill').fillna(0)
    
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all advanced features."""
        print("🔧 Engineering advanced features...")
        
        original_features = data.shape[1]
        
        # Create all feature sets
        vol_features = self.create_volatility_regime_features(data)
        momentum_features = self.create_cross_asset_momentum(data)
        temporal_features = self.create_temporal_pattern_features(data)
        correlation_features = self.create_correlation_cluster_features(data)
        
        # Combine all features
        enhanced_data = pd.concat([
            data, vol_features, momentum_features, 
            temporal_features, correlation_features
        ], axis=1)
        
        # Fill NaN values
        enhanced_data = enhanced_data.fillna(method='ffill').fillna(0)
        
        new_features = enhanced_data.shape[1] - original_features
        print(f"   Added {new_features} new features ({original_features} → {enhanced_data.shape[1]})")
        
        return enhanced_data

# ========================= ENSEMBLE STRATEGIES =========================

class EnsembleStrategy:
    """Advanced ensemble strategies for Sharpe optimization."""
    
    def __init__(self, base_model_class):
        self.base_model_class = base_model_class
        self.models = {}
        
    def train_diverse_models(self, X_train, y_train, X_val, y_val, device='cuda'):
        """Train models with different loss configurations."""
        print("🎯 Training diverse ensemble models...")
        
        model_configs = {
            'sharpe_focused': {'sharpe_weight': 0.9, 'mse_weight': 0.05, 'mae_weight': 0.05},
            'correlation_focused': {'sharpe_weight': 0.6, 'mse_weight': 0.3, 'mae_weight': 0.1},
            'stability_focused': {'sharpe_weight': 0.7, 'mse_weight': 0.15, 'mae_weight': 0.15},
            'mse_focused': {'sharpe_weight': 0.5, 'mse_weight': 0.4, 'mae_weight': 0.1}
        }
        
        for name, loss_config in model_configs.items():
            print(f"   Training {name} model...")
            
            model = self.base_model_class(X_train.shape[1], y_train.shape[1]).to(device)
            loss_fn = CombinedLoss(**loss_config)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Quick training (5 epochs for experiment)
            model.train()
            for epoch in range(5):
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()
            
            self.models[name] = model
            
        print(f"   Trained {len(self.models)} ensemble models")
        
    def predict_ensemble(self, X_test, weights=None):
        """Generate ensemble predictions."""
        if weights is None:
            weights = [1/len(self.models)] * len(self.models)  # Equal weights
            
        predictions = []
        for model in self.models.values():
            model.eval()
            with torch.no_grad():
                pred = model(X_test)
                predictions.append(pred)
        
        # Weighted average
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
        return ensemble_pred

# ========================= NEXT-GEN ARCHITECTURES =========================

class MultiHeadAttentionModel(nn.Module):
    """Next-generation model with multi-head attention."""
    
    def __init__(self, input_dim, n_targets=424):
        super().__init__()
        
        # Feature embedding
        self.embedding = nn.Linear(input_dim, 512)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Residual dense blocks
        self.dense_block1 = ResidualDenseBlock(512, 512)
        self.dense_block2 = ResidualDenseBlock(512, 256)
        self.dense_block3 = ResidualDenseBlock(256, 128)
        
        # Output layer
        self.output = nn.Linear(128, n_targets)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # Add sequence dimension for attention (batch_size, seq_len=1, features)
        embedded = embedded.unsqueeze(1)
        
        # Self-attention
        attended, _ = self.attention(embedded, embedded, embedded)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Residual dense blocks
        x = self.dense_block1(attended)
        x = self.dense_block2(x)
        x = self.dense_block3(x)
        
        # Output
        return self.output(x)

class ResidualDenseBlock(nn.Module):
    """Residual dense block with skip connections."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += identity
        out = self.activation(out)
        
        return out

# ========================= BASELINE MODEL (for comparison) =========================

class BaselineModel(nn.Module):
    """Current production model for baseline comparison."""
    
    def __init__(self, input_dim, n_targets=424):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, n_targets)
        )
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

# ========================= LOSS FUNCTIONS =========================

class CombinedLoss(nn.Module):
    """Combined loss function optimizing Sharpe-like score."""
    
    def __init__(self, sharpe_weight=0.7, mse_weight=0.2, mae_weight=0.1):
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.eps = 1e-8
    
    def pearson_correlation(self, x, y):
        """Calculate Pearson correlation."""
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        
        num = torch.sum(x_centered * y_centered)
        den = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
        
        return num / (den + self.eps)
    
    def forward(self, y_pred, y_true):
        batch_size, n_targets = y_pred.shape
        
        # Calculate correlations
        correlations = []
        for i in range(n_targets):
            corr = self.pearson_correlation(y_pred[:, i], y_true[:, i])
            correlations.append(torch.clamp(corr, -1.0 + self.eps, 1.0 - self.eps))
        
        correlations_tensor = torch.stack(correlations)
        mean_corr = torch.mean(correlations_tensor)
        std_corr = torch.std(correlations_tensor) + self.eps
        sharpe_like = mean_corr / std_corr
        
        # Auxiliary losses
        mse_loss = self.mse_loss(y_pred, y_true)
        mae_loss = self.mae_loss(y_pred, y_true)
        
        # Combined loss (negative sharpe for minimization)
        total_loss = (self.sharpe_weight * (-sharpe_like) + 
                     self.mse_weight * mse_loss + 
                     self.mae_weight * mae_loss)
        
        return total_loss

def calculate_sharpe_score(y_true, y_pred):
    """Calculate Sharpe-like score from predictions."""
    correlations = []
    
    for i in range(y_true.shape[1]):
        # Convert to numpy for correlation calculation
        true_vals = y_true[:, i].detach().cpu().numpy()
        pred_vals = y_pred[:, i].detach().cpu().numpy()
        
        # Remove NaN values
        mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
        if mask.sum() > 1:
            corr = np.corrcoef(true_vals[mask], pred_vals[mask])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    if len(correlations) == 0:
        return 0.0
        
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    if std_corr == 0:
        return 0.0
        
    return mean_corr / std_corr

# ========================= MAIN EXPERIMENT PIPELINE =========================

def run_ultra_prediction_experiments():
    """Run comprehensive prediction accuracy experiments."""
    
    print("🚀 ULTRA PREDICTION ACCURACY EXPERIMENTAL PIPELINE")
    print("="*60)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Load data
    print("📊 Loading data...")
    try:
        train_data = pd.read_csv('input/train.csv')
        train_labels = pd.read_csv('input/train_labels.csv')
        print(f"   Training data: {train_data.shape}")
        print(f"   Training labels: {train_labels.shape}")
    except FileNotFoundError:
        print("❌ Error: Data files not found. Make sure 'input/' directory exists.")
        return
    
    # Prepare data for experiments
    print("🔧 Preparing data...")
    
    # Remove date_id and align data
    if 'date_id' in train_data.columns:
        train_features = train_data.drop('date_id', axis=1)
    else:
        train_features = train_data
    
    if 'date_id' in train_labels.columns:
        target_features = train_labels.drop('date_id', axis=1)
    else:
        target_features = train_labels
    
    # Fill missing values
    train_features = train_features.fillna(method='ffill').fillna(0)
    target_features = target_features.fillna(method='ffill').fillna(0)
    
    # Align samples (ensure same number of rows)
    min_samples = min(len(train_features), len(target_features))
    train_features = train_features.iloc[:min_samples]
    target_features = target_features.iloc[:min_samples]
    
    print(f"   Aligned data: {train_features.shape[0]} samples, {train_features.shape[1]} features, {target_features.shape[1]} targets")
    
    # ===================== EXPERIMENT 1: BASELINE ======================
    
    print("\n🧪 EXPERIMENT 1: BASELINE MODEL")
    print("-" * 40)
    
    start_time = time.time()
    
    # Prepare baseline data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(train_features)
    y_scaled = scaler_y.fit_transform(target_features)
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Train/validation split (80/20)
    split_idx = int(0.8 * len(X_scaled))
    X_train = torch.FloatTensor(X_scaled[:split_idx]).to(device)
    y_train = torch.FloatTensor(y_scaled[:split_idx]).to(device)
    X_val = torch.FloatTensor(X_scaled[split_idx:]).to(device)
    y_val = torch.FloatTensor(y_scaled[split_idx:]).to(device)
    
    # Train baseline model
    baseline_model = BaselineModel(X_train.shape[1], y_train.shape[1]).to(device)
    loss_fn = CombinedLoss(sharpe_weight=0.7, mse_weight=0.2, mae_weight=0.1)
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    
    # Training loop
    baseline_model.train()
    for epoch in range(10):  # Quick training for experiment
        optimizer.zero_grad()
        y_pred = baseline_model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate baseline
    baseline_model.eval()
    with torch.no_grad():
        y_pred_val = baseline_model(X_val)
        baseline_sharpe = calculate_sharpe_score(y_val, y_pred_val)
    
    baseline_duration = (time.time() - start_time) / 60
    
    tracker.log_experiment(
        name="Baseline Model",
        sharpe=baseline_sharpe,
        metrics={'features': X_train.shape[1], 'targets': y_train.shape[1]},
        duration=baseline_duration
    )
    
    # ================= EXPERIMENT 2: ADVANCED FEATURES =================
    
    print("\n🧪 EXPERIMENT 2: ADVANCED FEATURE ENGINEERING")
    print("-" * 50)
    
    start_time = time.time()
    
    # Engineer advanced features
    feature_engineer = AdvancedFeatureEngineer()
    enhanced_features = feature_engineer.engineer_all_features(train_data)
    
    # Remove date_id if present
    if 'date_id' in enhanced_features.columns:
        enhanced_features = enhanced_features.drop('date_id', axis=1)
    
    # Align with targets
    enhanced_features = enhanced_features.iloc[:min_samples]
    
    # Scale enhanced features
    X_enhanced = scaler_X.fit_transform(enhanced_features)
    
    # Convert to tensors
    X_train_enh = torch.FloatTensor(X_enhanced[:split_idx]).to(device)
    X_val_enh = torch.FloatTensor(X_enhanced[split_idx:]).to(device)
    
    # Train enhanced model
    enhanced_model = BaselineModel(X_train_enh.shape[1], y_train.shape[1]).to(device)
    optimizer_enh = optim.Adam(enhanced_model.parameters(), lr=0.001)
    
    enhanced_model.train()
    for epoch in range(10):
        optimizer_enh.zero_grad()
        y_pred = enhanced_model(X_train_enh)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer_enh.step()
    
    # Evaluate enhanced model
    enhanced_model.eval()
    with torch.no_grad():
        y_pred_val_enh = enhanced_model(X_val_enh)
        enhanced_sharpe = calculate_sharpe_score(y_val, y_pred_val_enh)
    
    enhanced_duration = (time.time() - start_time) / 60
    
    tracker.log_experiment(
        name="Enhanced Features",
        sharpe=enhanced_sharpe,
        metrics={'features': X_train_enh.shape[1], 'new_features': X_train_enh.shape[1] - X_train.shape[1]},
        duration=enhanced_duration
    )
    
    # ================= EXPERIMENT 3: ENSEMBLE STRATEGY =================
    
    print("\n🧪 EXPERIMENT 3: ENSEMBLE STRATEGY")
    print("-" * 40)
    
    start_time = time.time()
    
    # Create ensemble
    ensemble = EnsembleStrategy(BaselineModel)
    ensemble.train_diverse_models(X_train, y_train, X_val, y_val, device)
    
    # Evaluate ensemble
    ensemble_pred = ensemble.predict_ensemble(X_val)
    ensemble_sharpe = calculate_sharpe_score(y_val, ensemble_pred)
    
    ensemble_duration = (time.time() - start_time) / 60
    
    tracker.log_experiment(
        name="Ensemble Strategy",
        sharpe=ensemble_sharpe,
        metrics={'models': len(ensemble.models), 'features': X_train.shape[1]},
        duration=ensemble_duration
    )
    
    # ================= EXPERIMENT 4: NEXT-GEN ARCHITECTURE =================
    
    print("\n🧪 EXPERIMENT 4: NEXT-GEN ARCHITECTURE")
    print("-" * 45)
    
    start_time = time.time()
    
    # Train attention model
    attention_model = MultiHeadAttentionModel(X_train.shape[1], y_train.shape[1]).to(device)
    optimizer_att = optim.Adam(attention_model.parameters(), lr=0.001)
    
    attention_model.train()
    for epoch in range(10):
        optimizer_att.zero_grad()
        y_pred = attention_model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer_att.step()
    
    # Evaluate attention model
    attention_model.eval()
    with torch.no_grad():
        y_pred_val_att = attention_model(X_val)
        attention_sharpe = calculate_sharpe_score(y_val, y_pred_val_att)
    
    attention_duration = (time.time() - start_time) / 60
    
    tracker.log_experiment(
        name="Multi-Head Attention",
        sharpe=attention_sharpe,
        metrics={'architecture': 'attention', 'heads': 8, 'features': X_train.shape[1]},
        duration=attention_duration
    )
    
    # ================= EXPERIMENT 5: ULTIMATE COMBINATION =================
    
    print("\n🧪 EXPERIMENT 5: ULTIMATE COMBINATION")
    print("-" * 45)
    
    start_time = time.time()
    
    # Enhanced features + Attention + Ensemble
    enhanced_ensemble = EnsembleStrategy(MultiHeadAttentionModel)
    enhanced_ensemble.train_diverse_models(X_train_enh, y_train, X_val_enh, y_val, device)
    
    ultimate_pred = enhanced_ensemble.predict_ensemble(X_val_enh)
    ultimate_sharpe = calculate_sharpe_score(y_val, ultimate_pred)
    
    ultimate_duration = (time.time() - start_time) / 60
    
    tracker.log_experiment(
        name="Ultimate Combination",
        sharpe=ultimate_sharpe,
        metrics={
            'features': X_train_enh.shape[1],
            'architecture': 'attention_ensemble', 
            'models': len(enhanced_ensemble.models)
        },
        duration=ultimate_duration
    )
    
    # ===================== FINAL RESULTS ======================
    
    print("\n🏆 EXPERIMENTAL RESULTS SUMMARY")
    print("="*60)
    
    results_df = tracker.get_summary()
    
    # Display results table
    print("\n📊 Results Table:")
    print("-" * 80)
    print(f"{'Experiment':<25} {'Sharpe':<10} {'Improvement':<12} {'Duration (min)':<15}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['name']:<25} {row['sharpe_score']:<10.4f} {row['improvement_pct']:>+8.1f}% {row['duration_minutes']:<15.2f}")
    
    print("-" * 80)
    
    # Find best performing experiment
    best_experiment = results_df.loc[results_df['sharpe_score'].idxmax()]
    
    print(f"\n🎯 BEST PERFORMER: {best_experiment['name']}")
    print(f"   Sharpe Score: {best_experiment['sharpe_score']:.4f}")
    print(f"   Improvement: {best_experiment['improvement_pct']:+.1f}%")
    print(f"   Duration: {best_experiment['duration_minutes']:.2f} minutes")
    
    # Save results
    results_file = 'results/experiments/ultra_prediction_results.json'
    print(f"\n💾 Saving results to {results_file}")
    
    # Convert results to JSON-serializable format
    results_json = {
        'experiment_summary': results_df.to_dict(orient='records'),
        'best_experiment': best_experiment.to_dict(),
        'baseline_sharpe': tracker.baseline_sharpe,
        'total_experiments': len(results_df),
        'total_duration_minutes': results_df['duration_minutes'].sum()
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("✅ Experimental pipeline completed!")
    
    return results_json

if __name__ == "__main__":
    results = run_ultra_prediction_experiments()