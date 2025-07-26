#!/usr/bin/env python3
"""
GPU Transformer Models for Track C - Advanced Feature Discovery
Implements PatchTST and TimeSeriesTransformer for Mitsui Challenge
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import math
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class PatchTST(nn.Module):
    """
    PatchTST: Patched Time Series Transformer
    Divides time series into patches and applies transformer attention
    """
    
    def __init__(self, input_dim, n_targets, patch_len=16, stride=8, 
                 d_model=128, nhead=8, num_layers=4, max_seq_len=1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # Output projection for each target
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(n_targets)
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def create_patches(self, x):
        """Create patches from input sequence."""
        batch_size, seq_len, features = x.shape
        
        # Calculate number of patches
        n_patches = (seq_len - self.patch_len) // self.stride + 1
        
        if n_patches <= 0:
            # If sequence too short, use the entire sequence as one patch
            patches = x.reshape(batch_size, 1, -1)
        else:
            patches = []
            for i in range(n_patches):
                start_idx = i * self.stride
                end_idx = start_idx + self.patch_len
                patch = x[:, start_idx:end_idx, :].reshape(batch_size, -1)
                patches.append(patch)
            patches = torch.stack(patches, dim=1)
        
        return patches
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape for time series processing (batch, seq, features)
        if len(x.shape) == 2:
            # If input is (batch, features), treat as single time step
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # Create patches
        patches = self.create_patches(x)  # (batch, n_patches, patch_len * features)
        n_patches = patches.shape[1]
        
        # Embed patches
        patch_embeddings = self.patch_embedding(patches)  # (batch, n_patches, d_model)
        
        # Add positional encoding
        patch_embeddings = self.pos_encoder(patch_embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(patch_embeddings)  # (batch, n_patches, d_model)
        
        # Global pooling across patches
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (batch, d_model)
        
        # Generate outputs for each target
        outputs = []
        for head in self.output_heads:
            outputs.append(head(pooled))
        
        return torch.cat(outputs, dim=1)

class TimeSeriesTransformer(nn.Module):
    """
    Standard Time Series Transformer with temporal attention
    """
    
    def __init__(self, input_dim, n_targets, d_model=128, nhead=8, 
                 num_layers=4, seq_len=50, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets)
        )
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Expand to sequence if needed
        if len(x.shape) == 2:
            # Replicate features to create temporal dimension
            x = x.unsqueeze(1).repeat(1, min(self.seq_len, 10), 1)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(x)  # (batch, seq, d_model)
        
        # Temporal pooling
        pooled = self.temporal_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (batch, d_model)
        
        # Output projection
        outputs = self.output_projection(pooled)
        
        return outputs

class AdvancedFeatureTransformer(nn.Module):
    """
    Advanced transformer that combines multiple attention mechanisms
    for sophisticated feature discovery
    """
    
    def __init__(self, input_dim, n_targets, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.d_model = d_model
        
        # Multi-scale feature extraction
        self.feature_projections = nn.ModuleList([
            nn.Linear(input_dim, d_model // 4),
            nn.Linear(input_dim, d_model // 4),
            nn.Linear(input_dim, d_model // 4),
            nn.Linear(input_dim, d_model // 4)
        ])
        
        # Multi-head self-attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(num_layers)
        ])
        
        # Cross-target attention
        self.cross_target_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=0.1, batch_first=True
        )
        
        # Output heads with target-specific processing
        self.target_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1)
            )
            for _ in range(n_targets)
        ])
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Multi-scale feature extraction
        features = []
        for proj in self.feature_projections:
            features.append(proj(x))
        
        # Concatenate multi-scale features
        x = torch.cat(features, dim=-1)  # (batch, d_model)
        x = x.unsqueeze(1)  # (batch, 1, d_model) for attention
        
        # Apply attention blocks
        for i, (attn, norm, ff) in enumerate(zip(self.attention_blocks, self.layer_norms, self.feed_forwards)):
            # Self-attention
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
            
            # Feed-forward
            ff_out = ff(x)
            x = norm(x + ff_out)
        
        # Cross-target attention
        cross_attn_out, _ = self.cross_target_attention(x, x, x)
        x = x + cross_attn_out
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch, d_model)
        
        # Target-specific processing
        outputs = []
        for processor in self.target_processors:
            outputs.append(processor(x))
        
        return torch.cat(outputs, dim=1)

class GPUTransformerExperimentRunner:
    """Runner for GPU transformer experiments."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        print(f"🤖 GPU Transformer Runner - Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    def manual_correlation(self, x, y):
        """Manual correlation calculation to avoid NumPy issues."""
        if torch.is_tensor(x):
            x_vals = x.cpu().tolist()
        else:
            x_vals = x.tolist() if hasattr(x, 'tolist') else list(x)
            
        if torch.is_tensor(y):
            y_vals = y.cpu().tolist()
        else:
            y_vals = y.tolist() if hasattr(y, 'tolist') else list(y)
        
        n = len(x_vals)
        if n == 0:
            return 0.0
            
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x_vals[i] * y_vals[i] for i in range(n))
        sum_x2 = sum(x * x for x in x_vals)
        sum_y2 = sum(y * y for y in y_vals)
        
        num = n * sum_xy - sum_x * sum_y
        den = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))**0.5
        
        return num / den if den != 0 else 0.0
    
    def calculate_sharpe_score(self, y_true, y_pred):
        """Calculate Sharpe-like score using manual correlation."""
        if torch.is_tensor(y_true):
            y_true = y_true.cpu()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu()
            
        n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
        correlations = []
        
        for i in range(n_targets):
            if len(y_true.shape) > 1:
                true_col = y_true[:, i]
                pred_col = y_pred[:, i]
            else:
                true_col = y_true
                pred_col = y_pred
                
            corr = self.manual_correlation(true_col, pred_col)
            if abs(corr) < 1.0:
                correlations.append(corr)
        
        if len(correlations) == 0:
            return 0.0, 0.0, 0.0
            
        mean_corr = sum(correlations) / len(correlations)
        std_corr = (sum((c - mean_corr)**2 for c in correlations) / len(correlations))**0.5
        sharpe_score = mean_corr / std_corr if std_corr > 0 else mean_corr
        
        return mean_corr, std_corr, sharpe_score
    
    def train_transformer_model(self, model, X_train, y_train, X_test, y_test, 
                              model_name="Transformer", epochs=30, lr=0.001):
        """Train a transformer model."""
        print(f"\n🤖 Training {model_name}...")
        
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Training data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"   Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
        
        mean_corr, std_corr, sharpe = self.calculate_sharpe_score(y_test_tensor, predictions)
        
        print(f"✅ {model_name} - Sharpe: {sharpe:.4f}, Time: {training_time:.2f}s")
        
        return {
            'model': model,
            'predictions': predictions,
            'sharpe_score': sharpe,
            'mean_corr': mean_corr,
            'std_corr': std_corr,
            'training_time': training_time
        }
    
    def run_transformer_experiments(self, X, y, experiment_name="GPU_Transformers"):
        """Run comprehensive transformer experiments."""
        print(f"\n🤖 {experiment_name} - Track C Feature Discovery")
        print("=" * 70)
        
        # Prepare data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        split_idx = int(0.75 * len(X_scaled))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
        print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        input_dim = X_train.shape[1]
        n_targets = y_train.shape[1]
        
        results = {}
        
        # 1. PatchTST (Fixed dimension issue)
        try:
            patch_len = min(4, max(2, input_dim // 4))  # Smaller patch size
            patch_tst = PatchTST(
                input_dim=input_dim, 
                n_targets=n_targets,
                patch_len=patch_len,
                stride=patch_len // 2,
                d_model=64,  # Smaller model for compatibility
                nhead=4,
                num_layers=2
            )
            results['PatchTST'] = self.train_transformer_model(
                patch_tst, X_train, y_train, X_test, y_test, "PatchTST", epochs=25
            )
        except Exception as e:
            print(f"❌ PatchTST failed: {e}")
        
        # 2. TimeSeriesTransformer
        try:
            ts_transformer = TimeSeriesTransformer(
                input_dim=input_dim,
                n_targets=n_targets,
                d_model=128,
                nhead=8,
                num_layers=4
            )
            results['TimeSeriesTransformer'] = self.train_transformer_model(
                ts_transformer, X_train, y_train, X_test, y_test, "TimeSeriesTransformer", epochs=25
            )
        except Exception as e:
            print(f"❌ TimeSeriesTransformer failed: {e}")
        
        # 3. AdvancedFeatureTransformer
        try:
            advanced_transformer = AdvancedFeatureTransformer(
                input_dim=input_dim,
                n_targets=n_targets,
                d_model=256,
                nhead=8,
                num_layers=4
            )
            results['AdvancedFeatureTransformer'] = self.train_transformer_model(
                advanced_transformer, X_train, y_train, X_test, y_test, "AdvancedFeatureTransformer", epochs=20
            )
        except Exception as e:
            print(f"❌ AdvancedFeatureTransformer failed: {e}")
        
        # Results summary
        print(f"\n📊 Track C Transformer Results")
        print("=" * 70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_score'], reverse=True)
        
        for i, (method, metrics) in enumerate(sorted_results):
            print(f"{i+1}. {method}")
            print(f"   Sharpe Score: {metrics['sharpe_score']:.4f}")
            print(f"   Mean Corr: {metrics['mean_corr']:.4f}")
            print(f"   Std Corr: {metrics['std_corr']:.4f}")
            print(f"   Training Time: {metrics['training_time']:.2f}s")
            print()
        
        # Save results
        results_data = []
        for method, metrics in results.items():
            results_data.append({
                'method': method,
                'sharpe_score': metrics['sharpe_score'],
                'mean_correlation': metrics['mean_corr'],
                'std_correlation': metrics['std_corr'],
                'training_time': metrics['training_time']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('GPU_TRANSFORMER_TRACK_C_RESULTS.csv', index=False)
        print(f"💾 Results saved to GPU_TRANSFORMER_TRACK_C_RESULTS.csv")
        
        # GPU memory usage
        if torch.cuda.is_available():
            print(f"\n⚡ GPU Memory Usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
            print(f"   Cached: {torch.cuda.memory_reserved(0) / (1024**2):.1f} MB")
        
        self.results[experiment_name] = results
        return results

def main():
    """Main execution for GPU transformer experiments."""
    print("🤖 GPU Transformer Models - Track C Feature Discovery")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️ Warning: GPU not available, using CPU")
    else:
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
    
    # Load data
    try:
        print("\n📁 Loading competition data...")
        train_data = pd.read_csv('input/train.csv').head(300)
        label_data = pd.read_csv('input/train_labels.csv').head(300)
        merged = train_data.merge(label_data, on='date_id', how='inner')
        
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:25]
        target_cols = [col for col in label_data.columns if col.startswith('target_')][:12]
        
        X = merged[feature_cols].fillna(0).values
        y = merged[target_cols].fillna(0).values
        
        print(f"✅ Data loaded: {X.shape}")
        
        # Run transformer experiments
        runner = GPUTransformerExperimentRunner()
        results = runner.run_transformer_experiments(X, y)
        
        print("\n🎉 GPU Transformer Experiments Completed!")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()