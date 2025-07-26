#!/usr/bin/env python3
"""
GPU-Optimized Sharpe-like Loss Function for Mitsui Challenge
Implements differentiable Sharpe-like loss for direct competition metric optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class GPUSharpelikeLoss(nn.Module):
    """
    GPU-optimized Sharpe-like loss function for the Mitsui competition.
    
    Competition metric: mean(Spearman correlations) / std(Spearman correlations)
    
    This implementation provides several differentiable approximations:
    1. Pearson correlation as Spearman approximation
    2. Rank-based differentiable correlation
    3. Smoothed ranking functions
    """
    
    def __init__(self, method='pearson', epsilon=1e-8, temperature=1.0, 
                 ranking_method='soft', penalty_weight=0.1):
        super().__init__()
        
        self.method = method
        self.epsilon = epsilon
        self.temperature = temperature
        self.ranking_method = ranking_method
        self.penalty_weight = penalty_weight
        
        print(f"🎯 GPU Sharpe-like Loss initialized:")
        print(f"   Method: {method}")
        print(f"   Ranking: {ranking_method}")
        print(f"   Temperature: {temperature}")
    
    def soft_ranking(self, x):
        """Soft differentiable ranking using temperature-scaled softmax."""
        # Create pairwise differences
        n = x.shape[0]
        x_expanded = x.unsqueeze(1)  # (n, 1)
        x_tiled = x.unsqueeze(0)     # (1, n)
        
        # Pairwise comparisons
        comparisons = (x_expanded - x_tiled) / self.temperature  # (n, n)
        
        # Soft ranking via cumulative distribution
        soft_ranks = torch.sum(torch.sigmoid(comparisons), dim=1)
        
        return soft_ranks
    
    def pearson_correlation(self, x, y):
        """Compute Pearson correlation coefficient (differentiable)."""
        # Center the variables
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        
        # Compute correlation
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
        
        correlation = numerator / (denominator + self.epsilon)
        return torch.clamp(correlation, -1.0 + self.epsilon, 1.0 - self.epsilon)
    
    def spearman_correlation_soft(self, x, y):
        """Soft differentiable approximation of Spearman correlation."""
        if self.ranking_method == 'soft':
            # Use soft ranking
            x_ranks = self.soft_ranking(x)
            y_ranks = self.soft_ranking(y)
        else:
            # Use hard ranking (not differentiable, but can be used for comparison)
            x_ranks = torch.argsort(torch.argsort(x)).float()
            y_ranks = torch.argsort(torch.argsort(y)).float()
        
        return self.pearson_correlation(x_ranks, y_ranks)
    
    def forward(self, y_pred, y_true):
        """
        Compute negative Sharpe-like score for loss minimization.
        
        Args:
            y_pred: Predicted values (batch_size, n_targets)
            y_true: True values (batch_size, n_targets)
            
        Returns:
            Negative Sharpe-like score (for minimization)
        """
        batch_size, n_targets = y_pred.shape
        
        if n_targets == 1:
            # Single target case
            if self.method == 'pearson':
                correlation = self.pearson_correlation(y_pred.squeeze(), y_true.squeeze())
            else:  # spearman
                correlation = self.spearman_correlation_soft(y_pred.squeeze(), y_true.squeeze())
            
            # Return negative correlation (we want to maximize correlation)
            return -correlation
        
        # Multi-target case
        correlations = []
        
        for i in range(n_targets):
            pred_i = y_pred[:, i]
            true_i = y_true[:, i]
            
            if self.method == 'pearson':
                corr = self.pearson_correlation(pred_i, true_i)
            else:  # spearman
                corr = self.spearman_correlation_soft(pred_i, true_i)
            
            correlations.append(corr)
        
        # Stack correlations
        correlations_tensor = torch.stack(correlations)
        
        # Compute Sharpe-like ratio
        mean_corr = torch.mean(correlations_tensor)
        std_corr = torch.std(correlations_tensor) + self.epsilon
        
        sharpe_like_score = mean_corr / std_corr
        
        # Add penalty for extreme correlations (regularization)
        extreme_penalty = self.penalty_weight * torch.mean(torch.abs(correlations_tensor))
        
        # Return negative Sharpe-like score for minimization
        return -(sharpe_like_score - extreme_penalty)

class AdaptiveSharpelikeLoss(nn.Module):
    """
    Adaptive Sharpe-like loss that adjusts during training.
    """
    
    def __init__(self, initial_temp=1.0, temp_decay=0.99, min_temp=0.1):
        super().__init__()
        
        self.initial_temp = initial_temp
        self.temp_decay = temp_decay
        self.min_temp = min_temp
        self.current_temp = initial_temp
        self.iteration = 0
        
        self.base_loss = GPUSharpelikeLoss(
            method='spearman', 
            temperature=self.current_temp,
            ranking_method='soft'
        )
    
    def update_temperature(self):
        """Update temperature for annealing."""
        self.iteration += 1
        self.current_temp = max(
            self.min_temp,
            self.initial_temp * (self.temp_decay ** self.iteration)
        )
        self.base_loss.temperature = self.current_temp
    
    def forward(self, y_pred, y_true):
        """Forward pass with temperature annealing."""
        self.update_temperature()
        return self.base_loss(y_pred, y_true)

class SharpelikeLossWithAuxiliary(nn.Module):
    """
    Sharpe-like loss combined with auxiliary losses for better training stability.
    """
    
    def __init__(self, sharpe_weight=0.7, mse_weight=0.2, mae_weight=0.1):
        super().__init__()
        
        self.sharpe_weight = sharpe_weight
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        
        self.sharpe_loss = GPUSharpelikeLoss(method='pearson')
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, y_pred, y_true):
        """Compute combined loss."""
        sharpe_loss = self.sharpe_loss(y_pred, y_true)
        mse_loss = self.mse_loss(y_pred, y_true)
        mae_loss = self.mae_loss(y_pred, y_true)
        
        total_loss = (self.sharpe_weight * sharpe_loss + 
                     self.mse_weight * mse_loss + 
                     self.mae_weight * mae_loss)
        
        return total_loss

class SharpeOptimizedModel(nn.Module):
    """Neural network specifically designed for Sharpe-like optimization."""
    
    def __init__(self, input_dim, n_targets, hidden_dims=[128, 64], 
                 dropout=0.1, use_batch_norm=True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_targets))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize with Xavier/Glorot initialization for better Sharpe optimization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def train_with_sharpe_loss(model, train_loader, val_loader, device, 
                          loss_function, epochs=50, lr=0.001):
    """Train model with Sharpe-like loss function."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"🚀 Training with Sharpe-like loss...")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = loss_function(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"   Early stopping at epoch {epoch}")
                break
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                  f"Val Loss = {avg_val_loss:.6f}")
    
    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def compare_loss_functions(X, y, test_size=0.2):
    """Compare different Sharpe-like loss implementations."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎯 Comparing Sharpe-like Loss Functions")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Prepare data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int((1 - test_size) * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = X_train.shape[1]
    n_targets = y_train.shape[1]
    
    # Define loss functions to compare
    loss_functions = {
        'Pearson_Sharpe': GPUSharpelikeLoss(method='pearson'),
        'Spearman_Soft': GPUSharpelikeLoss(method='spearman', ranking_method='soft'),
        'Adaptive_Sharpe': AdaptiveSharpelikeLoss(),
        'Combined_Loss': SharpelikeLossWithAuxiliary(),
        'MSE_Baseline': nn.MSELoss()
    }
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n🧪 Testing {loss_name}...")
        
        # Create fresh model
        model = SharpeOptimizedModel(input_dim, n_targets).to(device)
        
        start_time = time.time()
        
        # Train model
        train_losses, val_losses = train_with_sharpe_loss(
            model, train_loader, test_loader, device, loss_fn, epochs=20
        )
        
        training_time = time.time() - start_time
        
        # Evaluate final performance
        model.eval()
        total_test_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                total_test_loss += loss_fn(outputs, batch_y).item()
                
                all_predictions.append(outputs.cpu())
                all_targets.append(batch_y.cpu())
        
        # Calculate true Sharpe-like score
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Manual Sharpe calculation using manual correlation
        def manual_correlation(x, y):
            """Manual correlation avoiding NumPy."""
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
        
        correlations = []
        for i in range(n_targets):
            corr = manual_correlation(predictions[:, i], targets[:, i])
            if abs(corr) < 1.0:  # Filter out perfect correlations
                correlations.append(corr)
        
        if len(correlations) > 0:
            mean_corr = sum(correlations) / len(correlations)
            std_corr = (sum((c - mean_corr)**2 for c in correlations) / len(correlations))**0.5
            sharpe_score = mean_corr / std_corr if std_corr > 0 else mean_corr
        else:
            mean_corr = std_corr = sharpe_score = 0.0
        
        results[loss_name] = {
            'final_test_loss': total_test_loss / len(test_loader),
            'training_time': training_time,
            'sharpe_score': sharpe_score,
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'final_train_loss': train_losses[-1],
            'convergence_epochs': len(train_losses)
        }
        
        print(f"✅ {loss_name} completed:")
        print(f"   Sharpe score: {sharpe_score:.4f}")
        print(f"   Training time: {training_time:.2f}s")
    
    # Results summary
    print(f"\n📊 Loss Function Comparison Results")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_score'], reverse=True)
    
    for i, (loss_name, metrics) in enumerate(sorted_results):
        print(f"{i+1}. {loss_name}")
        print(f"   Sharpe Score: {metrics['sharpe_score']:.4f}")
        print(f"   Mean Correlation: {metrics['mean_correlation']:.4f}")
        print(f"   Training Time: {metrics['training_time']:.2f}s")
        print(f"   Convergence: {metrics['convergence_epochs']} epochs")
        print()
    
    # Save results
    results_df = pd.DataFrame([
        {
            'loss_function': name,
            'sharpe_score': metrics['sharpe_score'],
            'mean_correlation': metrics['mean_correlation'],
            'std_correlation': metrics['std_correlation'],
            'training_time': metrics['training_time'],
            'convergence_epochs': metrics['convergence_epochs']
        }
        for name, metrics in results.items()
    ])
    
    results_df.to_csv('GPU_SHARPE_LOSS_COMPARISON.csv', index=False)
    print(f"💾 Results saved to GPU_SHARPE_LOSS_COMPARISON.csv")
    
    return results

def main():
    """Main execution for Sharpe-like loss experiments."""
    print("🎯 GPU-Optimized Sharpe-like Loss Functions")
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
        train_data = pd.read_csv('input/train.csv').head(200)
        label_data = pd.read_csv('input/train_labels.csv').head(200)
        merged = train_data.merge(label_data, on='date_id', how='inner')
        
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:15]
        target_cols = [col for col in label_data.columns if col.startswith('target_')][:6]
        
        X = merged[feature_cols].fillna(0).values
        y = merged[target_cols].fillna(0).values
        
        print(f"✅ Data loaded: {X.shape}")
        
        # Run loss function comparison
        results = compare_loss_functions(X, y)
        
        # GPU memory usage
        if torch.cuda.is_available():
            print(f"\n⚡ GPU Memory Usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
            print(f"   Cached: {torch.cuda.memory_reserved(0) / (1024**2):.1f} MB")
        
        print("\n🎉 Sharpe-like Loss Function Experiments Completed!")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()