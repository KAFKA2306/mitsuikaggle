#!/usr/bin/env python3
"""
Production GPU-Accelerated Ensemble for Mitsui Challenge
Implements high-performance ensemble with GPU neural networks + CPU boosting models
Avoids NumPy compatibility issues by using workarounds
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

class ProductionGPUEnsemble:
    """Production-ready GPU ensemble avoiding NumPy compatibility issues."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        print(f"🚀 Production GPU Ensemble - Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    def manual_correlation(self, x, y):
        """Manual correlation calculation to avoid NumPy issues."""
        # Convert tensors to Python lists to avoid numpy conversion
        if torch.is_tensor(x):
            x_vals = x.cpu().tolist()
        else:
            x_vals = x.tolist() if hasattr(x, 'tolist') else list(x)
            
        if torch.is_tensor(y):
            y_vals = y.cpu().tolist()
        else:
            y_vals = y.tolist() if hasattr(y, 'tolist') else list(y)
        
        # Pearson correlation calculation
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
            if abs(corr) < 1.0:  # Filter out perfect/invalid correlations
                correlations.append(corr)
        
        if len(correlations) == 0:
            return 0.0, 0.0, 0.0
            
        mean_corr = sum(correlations) / len(correlations)
        std_corr = (sum((c - mean_corr)**2 for c in correlations) / len(correlations))**0.5
        sharpe_score = mean_corr / std_corr if std_corr > 0 else mean_corr
        
        return mean_corr, std_corr, sharpe_score

class GPUNeuralModel(nn.Module):
    """Optimized GPU neural network for multi-target regression."""
    
    def __init__(self, input_dim, n_targets, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Target-specific output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in range(n_targets)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared = self.shared_layers(x)
        outputs = []
        for head in self.output_heads:
            outputs.append(head(shared))
        return torch.cat(outputs, dim=1)

def run_production_experiment(n_samples=200, n_features=20, n_targets=10):
    """Run production GPU ensemble experiment."""
    
    print(f"\n🚀 Production GPU Ensemble Experiment")
    print("=" * 60)
    print(f"Samples: {n_samples}, Features: {n_features}, Targets: {n_targets}")
    
    # Initialize ensemble
    ensemble = ProductionGPUEnsemble()
    
    # Load and prepare data
    print("\n📁 Loading data...")
    try:
        train_data = pd.read_csv('input/train.csv').head(n_samples)
        label_data = pd.read_csv('input/train_labels.csv').head(n_samples)
        merged = train_data.merge(label_data, on='date_id', how='inner')
        
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:n_features]
        target_cols = [col for col in label_data.columns if col.startswith('target_')][:n_targets]
        
        X = merged[feature_cols].fillna(0).values
        y = merged[target_cols].fillna(0).values
        
        print(f"✅ Data loaded: {X.shape}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None
    
    # Data preprocessing
    X_scaled = ensemble.scaler.fit_transform(X)
    split_idx = int(0.75 * len(X_scaled))
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✅ Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Convert to GPU tensors
    X_train_gpu = torch.FloatTensor(X_train).to(ensemble.device)
    X_test_gpu = torch.FloatTensor(X_test).to(ensemble.device)
    y_train_gpu = torch.FloatTensor(y_train).to(ensemble.device)
    y_test_gpu = torch.FloatTensor(y_test).to(ensemble.device)
    
    results = {}
    
    # 1. GPU Neural Network
    print(f"\n🚀 Training GPU Neural Network...")
    try:
        start_time = time.time()
        
        model = GPUNeuralModel(X_train.shape[1], y_train.shape[1]).to(ensemble.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_gpu)
            loss = criterion(outputs, y_train_gpu)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            neural_predictions = model(X_test_gpu)
        
        training_time = time.time() - start_time
        mean_corr, std_corr, sharpe = ensemble.calculate_sharpe_score(y_test_gpu, neural_predictions)
        
        results['GPU_Neural'] = {
            'sharpe_score': sharpe,
            'mean_corr': mean_corr,
            'std_corr': std_corr,
            'time': training_time
        }
        
        print(f"✅ GPU Neural: Sharpe={sharpe:.4f}, Time={training_time:.2f}s")
        
    except Exception as e:
        print(f"❌ GPU Neural failed: {e}")
    
    # 2. XGBoost GPU
    print(f"\n⚡ Training XGBoost GPU...")
    try:
        start_time = time.time()
        
        xgb_predictions = []
        for i in range(y_train.shape[1]):
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                tree_method='gpu_hist',
                gpu_id=0,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train[:, i])
            pred = xgb_model.predict(X_test)
            xgb_predictions.append(pred)
        
        # Convert to tensor for evaluation
        xgb_pred_tensor = torch.FloatTensor([xgb_predictions]).squeeze().T.to(ensemble.device)
        
        training_time = time.time() - start_time
        mean_corr, std_corr, sharpe = ensemble.calculate_sharpe_score(y_test_gpu, xgb_pred_tensor)
        
        results['XGBoost_GPU'] = {
            'sharpe_score': sharpe,
            'mean_corr': mean_corr,
            'std_corr': std_corr,
            'time': training_time
        }
        
        print(f"✅ XGBoost GPU: Sharpe={sharpe:.4f}, Time={training_time:.2f}s")
        
    except Exception as e:
        print(f"❌ XGBoost GPU failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. LightGBM CPU
    print(f"\n💻 Training LightGBM CPU...")
    try:
        start_time = time.time()
        
        lgb_predictions = []
        for i in range(y_train.shape[1]):
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                device='cpu',
                verbose=-1
            )
            lgb_model.fit(X_train, y_train[:, i])
            pred = lgb_model.predict(X_test)
            lgb_predictions.append(pred)
        
        # Convert to tensor for evaluation
        lgb_pred_tensor = torch.FloatTensor([lgb_predictions]).squeeze().T.to(ensemble.device)
        
        training_time = time.time() - start_time
        mean_corr, std_corr, sharpe = ensemble.calculate_sharpe_score(y_test_gpu, lgb_pred_tensor)
        
        results['LightGBM_CPU'] = {
            'sharpe_score': sharpe,
            'mean_corr': mean_corr,
            'std_corr': std_corr,
            'time': training_time
        }
        
        print(f"✅ LightGBM CPU: Sharpe={sharpe:.4f}, Time={training_time:.2f}s")
        
    except Exception as e:
        print(f"❌ LightGBM failed: {e}")
    
    # 4. GPU Ensemble (equal weights)
    print(f"\n🎯 Creating GPU Ensemble...")
    try:
        ensemble_pred = (neural_predictions + xgb_pred_tensor + lgb_pred_tensor) / 3
        mean_corr, std_corr, sharpe = ensemble.calculate_sharpe_score(y_test_gpu, ensemble_pred)
        
        results['GPU_Ensemble'] = {
            'sharpe_score': sharpe,
            'mean_corr': mean_corr,
            'std_corr': std_corr,
            'time': 0.0  # No additional training time
        }
        
        print(f"✅ GPU Ensemble: Sharpe={sharpe:.4f}")
        
    except Exception as e:
        print(f"❌ Ensemble failed: {e}")
    
    # Results summary
    print(f"\n📊 Final Results Summary")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_score'], reverse=True)
    
    for i, (method, metrics) in enumerate(sorted_results):
        print(f"{i+1}. {method}")
        print(f"   Sharpe Score: {metrics['sharpe_score']:.4f}")
        print(f"   Mean Corr: {metrics['mean_corr']:.4f}")
        print(f"   Std Corr: {metrics['std_corr']:.4f}")
        print(f"   Time: {metrics['time']:.2f}s")
        print()
    
    # Calculate improvement
    if len(sorted_results) >= 2:
        best_score = sorted_results[0][1]['sharpe_score']
        baseline_score = sorted_results[-1][1]['sharpe_score']
        if baseline_score > 0:
            improvement = ((best_score - baseline_score) / baseline_score) * 100
            print(f"🎯 Best Method: {sorted_results[0][0]}")
            print(f"📈 Improvement: {improvement:.1f}%")
    
    # Save results
    results_data = []
    for method, metrics in results.items():
        results_data.append({
            'method': method,
            'sharpe_score': metrics['sharpe_score'],
            'mean_correlation': metrics['mean_corr'],
            'std_correlation': metrics['std_corr'],
            'training_time': metrics['time']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('GPU_PRODUCTION_RESULTS.csv', index=False)
    print(f"\n💾 Results saved to GPU_PRODUCTION_RESULTS.csv")
    
    # GPU memory usage
    if torch.cuda.is_available():
        print(f"\n⚡ GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / (1024**2):.1f} MB")
    
    return results

def main():
    """Main execution function."""
    print("🚀 Production GPU-Accelerated Ensemble for Mitsui Challenge")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️ Warning: GPU not available, using CPU fallback")
    else:
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
    
    # Run experiments with different scales
    scales = [
        (200, 20, 10),   # Small scale test
        (400, 30, 15),   # Medium scale
        (600, 40, 20),   # Larger scale
    ]
    
    all_results = {}
    
    for i, (samples, features, targets) in enumerate(scales):
        print(f"\n{'='*20} Experiment {i+1}: {samples} samples {'='*20}")
        try:
            results = run_production_experiment(samples, features, targets)
            all_results[f"Experiment_{i+1}"] = results
        except Exception as e:
            print(f"❌ Experiment {i+1} failed: {e}")
            continue
    
    print(f"\n🎉 All GPU Production Experiments Completed!")
    return all_results

if __name__ == "__main__":
    main()