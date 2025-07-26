#!/usr/bin/env python3
"""
Simple GPU Test for Mitsui Challenge - Bypass NumPy compatibility issues
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

print("🚀 Simple GPU Test for Mitsui Challenge")
print("=" * 50)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

# Load data
print("\n📁 Loading data...")
train_data = pd.read_csv('input/train.csv').head(100)
label_data = pd.read_csv('input/train_labels.csv').head(100)
merged = train_data.merge(label_data, on='date_id', how='inner')

# Prepare features and targets
feature_cols = [col for col in train_data.columns if col != 'date_id'][:10]
target_cols = [col for col in label_data.columns if col.startswith('target_')][:3]

X = merged[feature_cols].fillna(0).values
y = merged[target_cols].fillna(0).values

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")

# Split data
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Test 1: Simple GPU computation
print("\n🧪 Test 1: Basic GPU computation")
try:
    x_gpu = torch.randn(1000, 100, device=device)
    y_gpu = torch.randn(1000, 100, device=device)
    z_gpu = torch.matmul(x_gpu, y_gpu.T)
    print("✅ GPU matrix multiplication successful")
except Exception as e:
    print(f"❌ GPU computation failed: {e}")

# Test 2: Simple Neural Network on GPU
print("\n🧪 Test 2: Neural Network on GPU")
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

try:
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create and train model
    model = SimpleNet(X_train.shape[1], y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training neural network...")
    start_time = time.time()
    
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    training_time = time.time() - start_time
    print(f"✅ GPU neural network trained in {training_time:.2f}s")
    
    # Get predictions (stay in PyTorch)
    model.eval()
    with torch.no_grad():
        gpu_predictions = model(X_test_tensor)
        
    print(f"✅ GPU predictions shape: {gpu_predictions.shape}")
    
except Exception as e:
    print(f"❌ Neural network failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: XGBoost GPU
print("\n🧪 Test 3: XGBoost GPU")
try:
    gpu_xgb_predictions = []
    start_time = time.time()
    
    for i in range(y_train.shape[1]):
        xgb_model = xgb.XGBRegressor(
            n_estimators=50,
            tree_method='gpu_hist',
            gpu_id=0,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train[:, i])
        pred = xgb_model.predict(X_test)
        gpu_xgb_predictions.append(pred)
    
    training_time = time.time() - start_time
    print(f"✅ XGBoost GPU trained in {training_time:.2f}s")
    
except Exception as e:
    print(f"❌ XGBoost GPU failed: {e}")

# Test 4: LightGBM CPU (fallback)
print("\n🧪 Test 4: LightGBM CPU")
try:
    cpu_lgb_predictions = []
    start_time = time.time()
    
    for i in range(y_train.shape[1]):
        lgb_model = lgb.LGBMRegressor(
            n_estimators=50,
            device='cpu',
            verbose=-1
        )
        lgb_model.fit(X_train, y_train[:, i])
        pred = lgb_model.predict(X_test)
        cpu_lgb_predictions.append(pred)
    
    training_time = time.time() - start_time
    print(f"✅ LightGBM CPU trained in {training_time:.2f}s")
    
except Exception as e:
    print(f"❌ LightGBM failed: {e}")

# Test 5: Manual Sharpe-like calculation (avoid scipy)
print("\n🧪 Test 5: Manual Sharpe-like calculation")
try:
    def manual_spearman_approx(x, y):
        """Simple correlation approximation to avoid scipy issues."""
        # Convert to numpy manually using Python list conversion
        if torch.is_tensor(x):
            x_vals = x.cpu().tolist()
        else:
            x_vals = x.tolist() if hasattr(x, 'tolist') else list(x)
            
        if torch.is_tensor(y):
            y_vals = y.cpu().tolist()
        else:
            y_vals = y.tolist() if hasattr(y, 'tolist') else list(y)
        
        # Simple correlation calculation
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x_vals[i] * y_vals[i] for i in range(n))
        sum_x2 = sum(x * x for x in x_vals)
        sum_y2 = sum(y * y for y in y_vals)
        
        num = n * sum_xy - sum_x * sum_y
        den = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))**0.5
        
        return num / den if den != 0 else 0
    
    # Test with GPU predictions
    if 'gpu_predictions' in locals():
        correlations = []
        for i in range(y_test.shape[1]):
            corr = manual_spearman_approx(
                gpu_predictions[:, i], 
                torch.FloatTensor(y_test[:, i])
            )
            correlations.append(corr)
        
        mean_corr = sum(correlations) / len(correlations)
        std_corr = (sum((c - mean_corr)**2 for c in correlations) / len(correlations))**0.5
        sharpe_score = mean_corr / std_corr if std_corr > 0 else mean_corr
        
        print(f"✅ GPU Neural Network Results:")
        print(f"   Mean correlation: {mean_corr:.4f}")
        print(f"   Std correlation: {std_corr:.4f}")
        print(f"   Sharpe-like score: {sharpe_score:.4f}")

except Exception as e:
    print(f"❌ Manual calculation failed: {e}")

print("\n🎉 GPU Testing Completed!")
print(f"✅ GPU Available: {torch.cuda.is_available()}")
print(f"✅ PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ GPU Memory Used: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
    print(f"✅ GPU Memory Cached: {torch.cuda.memory_reserved(0) / (1024**2):.1f} MB")