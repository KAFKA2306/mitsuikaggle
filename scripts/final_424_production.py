#!/usr/bin/env python3
"""
Production 424 Targets Implementation
NumPy-free version using manual correlation calculations
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time
import warnings
warnings.filterwarnings('ignore')

class ProductionModel(nn.Module):
    """Production model for 424 targets."""
    
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

class ProductionLoss(nn.Module):
    """Production loss function avoiding NumPy dependencies."""
    
    def __init__(self, sharpe_weight=0.7, mse_weight=0.2, mae_weight=0.1):
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.eps = 1e-8
    
    def pearson_correlation(self, x, y):
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

def manual_correlation(x_vals, y_vals):
    """Manual correlation avoiding NumPy."""
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

def calculate_sharpe_score_manual(y_true, y_pred):
    """Calculate Sharpe score using manual correlation."""
    correlations = []
    
    for i in range(y_true.shape[1]):
        # Convert tensors to lists
        if torch.is_tensor(y_true):
            true_vals = y_true[:, i].cpu().tolist()
        else:
            true_vals = y_true[:, i].tolist() if hasattr(y_true[:, i], 'tolist') else list(y_true[:, i])
            
        if torch.is_tensor(y_pred):
            pred_vals = y_pred[:, i].cpu().tolist()
        else:
            pred_vals = y_pred[:, i].tolist() if hasattr(y_pred[:, i], 'tolist') else list(y_pred[:, i])
        
        corr = manual_correlation(true_vals, pred_vals)
        if abs(corr) < 1.0:  # Filter perfect correlations
            correlations.append(corr)
    
    if len(correlations) == 0:
        return 0.0, 0.0, 0.0
    
    mean_corr = sum(correlations) / len(correlations)
    var_corr = sum((c - mean_corr)**2 for c in correlations) / len(correlations)
    std_corr = var_corr**0.5
    sharpe_score = mean_corr / std_corr if std_corr > 0 else mean_corr
    
    return mean_corr, std_corr, sharpe_score

def simple_standardize(X):
    """Simple standardization without sklearn."""
    X_std = []
    means = []
    stds = []
    
    for col in range(X.shape[1]):
        col_data = X[:, col]
        mean_val = sum(col_data) / len(col_data)
        var_val = sum((x - mean_val)**2 for x in col_data) / (len(col_data) - 1)
        std_val = var_val**0.5 if var_val > 0 else 1.0
        
        means.append(mean_val)
        stds.append(std_val)
        
        standardized_col = [(x - mean_val) / std_val for x in col_data]
        X_std.append(standardized_col)
    
    # Transpose to get samples x features
    X_standardized = [[X_std[col][row] for col in range(len(X_std))] for row in range(len(X_std[0]))]
    
    return X_standardized, means, stds

def apply_standardization(X, means, stds):
    """Apply existing standardization parameters."""
    X_std = []
    
    for row in range(X.shape[0]):
        std_row = []
        for col in range(X.shape[1]):
            std_val = (X[row, col] - means[col]) / stds[col]
            std_row.append(std_val)
        X_std.append(std_row)
    
    return X_std

def prepare_data_simple():
    """Prepare data without sklearn dependencies."""
    print("📁 Loading dataset...")
    
    # Load data
    train_data = pd.read_csv('input/train.csv')
    label_data = pd.read_csv('input/train_labels.csv')
    
    print(f"   Train: {train_data.shape}, Labels: {label_data.shape}")
    
    # Merge
    merged = train_data.merge(label_data, on='date_id', how='inner')
    
    # Features and targets
    feature_cols = [col for col in train_data.columns if col != 'date_id']
    target_cols = [col for col in label_data.columns if col.startswith('target_')]
    
    print(f"   Features: {len(feature_cols)}, Targets: {len(target_cols)}")
    
    # Prepare arrays
    X = merged[feature_cols].fillna(method='ffill').fillna(0).values
    y = merged[target_cols].fillna(0).values
    
    # Simple train/test split (80/20)
    split_idx = int(0.8 * len(X))
    X_train_raw = X[:split_idx]
    X_test_raw = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Standardize
    X_train_std, means, stds = simple_standardize(X_train_raw)
    X_test_std = apply_standardization(X_test_raw, means, stds)
    
    print(f"   Train: {len(X_train_std)}, Test: {len(X_test_std)}")
    
    return X_train_std, X_test_std, y_train, y_test, means, stds

def train_production_model(X_train, y_train, X_test, y_test, device):
    """Train production model."""
    print(f"\n🚀 Training on {device}...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Model setup
    input_dim = X_train_tensor.shape[1]
    n_targets = y_train_tensor.shape[1]
    
    model = ProductionModel(input_dim, n_targets).to(device)
    criterion = ProductionLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_sharpe = float('-inf')
    patience_counter = 0
    
    for epoch in range(50):
        start_time = time.time()
        
        # Training
        model.train()
        
        # Simple batch processing
        batch_size = 32
        n_batches = (len(X_train_tensor) + batch_size - 1) // batch_size
        
        epoch_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train_tensor))
            
            batch_X = X_train_tensor[start_idx:end_idx]
            batch_y = y_train_tensor[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
            
            # Calculate Sharpe score
            mean_corr, std_corr, sharpe_score = calculate_sharpe_score_manual(
                y_test_tensor, test_outputs
            )
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping
            if sharpe_score > best_sharpe:
                best_sharpe = sharpe_score
                patience_counter = 0
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'sharpe_score': best_sharpe,
                    'mean_correlation': mean_corr,
                    'std_correlation': std_corr
                }, 'production_424_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"   Early stopping at epoch {epoch}")
                    break
            
            epoch_time = time.time() - start_time
            print(f"   Epoch {epoch:2d}: Loss={avg_loss:.6f}, Sharpe={sharpe_score:.4f}, Time={epoch_time:.1f}s")
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n🏆 Best Sharpe score: {best_sharpe:.4f}")
    
    # Load best model
    checkpoint = torch.load('production_424_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_sharpe

def generate_submission_simple(model, means, stds, device):
    """Generate submission without sklearn."""
    print("\n🔮 Generating submission...")
    
    try:
        # Load test data
        test_data = pd.read_csv('input/test.csv')
        print(f"   Test data: {test_data.shape}")
        
        # Prepare features
        feature_cols = [col for col in test_data.columns if col != 'date_id']
        X_test_raw = test_data[feature_cols].fillna(method='ffill').fillna(0).values
        
        # Apply standardization
        X_test_std = apply_standardization(X_test_raw, means, stds)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_std).to(device)
            predictions = model(X_tensor).cpu()
        
        # Create submission
        submission = pd.DataFrame({'date_id': test_data['date_id']})
        
        for i in range(424):
            pred_col = [float(predictions[j, i].item()) for j in range(predictions.shape[0])]
            submission[f'target_{i}'] = pred_col
        
        submission.to_csv('submission_production_424.csv', index=False)
        print(f"✅ Submission saved: submission_production_424.csv ({submission.shape})")
        
        return submission
        
    except Exception as e:
        print(f"❌ Submission failed: {e}")
        return None

def main():
    """Main production execution."""
    print("🎯 Production 424 Targets Implementation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, means, stds = prepare_data_simple()
        
        # Train model
        model, best_sharpe = train_production_model(X_train, y_train, X_test, y_test, device)
        
        # Final evaluation
        print(f"\n📊 Final evaluation...")
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            final_predictions = model(X_test_tensor)
        
        final_mean_corr, final_std_corr, final_sharpe = calculate_sharpe_score_manual(
            torch.FloatTensor(y_test), final_predictions
        )
        
        print(f"   Final Sharpe: {final_sharpe:.4f}")
        print(f"   Mean correlation: {final_mean_corr:.4f}")
        print(f"   Std correlation: {final_std_corr:.4f}")
        
        # Generate submission
        submission = generate_submission_simple(model, means, stds, device)
        
        # Save results
        results = {
            'experiment': 'Production_424_Targets',
            'samples': len(X_train) + len(X_test),
            'features': len(X_train[0]),
            'targets': y_train.shape[1],
            'best_sharpe_score': best_sharpe,
            'final_sharpe_score': final_sharpe,
            'final_mean_correlation': final_mean_corr,
            'final_std_correlation': final_std_corr,
            'total_time_minutes': (time.time() - start_time) / 60,
            'device': str(device)
        }
        
        with open('production_424_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Production Implementation Complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best Sharpe: {best_sharpe:.4f}")
        print(f"   Final Sharpe: {final_sharpe:.4f}")
        print(f"   Model: production_424_model.pth")
        print(f"   Submission: submission_production_424.csv")
        
        return results
        
    except Exception as e:
        print(f"❌ Production failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()