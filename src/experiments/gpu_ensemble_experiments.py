#!/usr/bin/env python3
"""
GPU-Accelerated Ensemble Experiments for Mitsui Commodity Challenge
Implements high-performance ensemble strategies using PyTorch CUDA acceleration.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

class GPUEnsembleConfig:
    """Configuration for GPU-accelerated ensemble experiments."""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 256
        self.learning_rate = 0.001
        self.epochs = 100
        self.early_stopping_patience = 10
        self.random_seed = 42
        
        # Ensemble weights optimization
        self.ensemble_lr = 0.01
        self.ensemble_epochs = 50
        
        print(f"🚀 GPU Ensemble Config - Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

class SharpeOptimizedLoss(nn.Module):
    """GPU-optimized Sharpe-like loss function for competition metric."""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        """
        Compute Sharpe-like loss: -mean(correlations) / std(correlations)
        Uses differentiable correlation approximation.
        """
        batch_size, n_targets = y_pred.shape
        correlations = []
        
        for i in range(n_targets):
            pred_i = y_pred[:, i]
            true_i = y_true[:, i]
            
            # Pearson correlation (differentiable approximation)
            mean_pred = torch.mean(pred_i)
            mean_true = torch.mean(true_i)
            
            num = torch.sum((pred_i - mean_pred) * (true_i - mean_true))
            den = torch.sqrt(torch.sum((pred_i - mean_pred) ** 2) * torch.sum((true_i - mean_true) ** 2))
            
            corr = num / (den + self.epsilon)
            correlations.append(corr)
        
        correlations = torch.stack(correlations)
        mean_corr = torch.mean(correlations)
        std_corr = torch.std(correlations) + self.epsilon
        
        # Return negative Sharpe-like score (for minimization)
        return -(mean_corr / std_corr)

class GPUNeuralEnsemble(nn.Module):
    """GPU-accelerated neural network ensemble for multi-target prediction."""
    def __init__(self, input_dim, n_targets, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_targets = n_targets
        
        # Shared bottom layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Target-specific heads
        self.target_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                nn.Linear(64, 1)
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
        shared_features = self.shared_layers(x)
        outputs = []
        
        for head in self.target_heads:
            outputs.append(head(shared_features))
        
        return torch.cat(outputs, dim=1)

class GPUEnsembleWeightOptimizer(nn.Module):
    """GPU-accelerated ensemble weight optimization."""
    def __init__(self, n_models, n_targets, constraint='simplex'):
        super().__init__()
        self.n_models = n_models
        self.n_targets = n_targets
        self.constraint = constraint
        
        # Learnable weights for each target and model
        if constraint == 'simplex':
            # Use softmax to ensure weights sum to 1
            self.raw_weights = nn.Parameter(torch.randn(n_targets, n_models))
        else:
            # Unconstrained weights
            self.weights = nn.Parameter(torch.ones(n_targets, n_models) / n_models)
    
    def forward(self):
        if self.constraint == 'simplex':
            return torch.softmax(self.raw_weights, dim=1)
        else:
            return torch.sigmoid(self.weights)

class GPUEnsembleExperimentRunner:
    """Main runner for GPU-accelerated ensemble experiments."""
    
    def __init__(self, config=None):
        self.config = config or GPUEnsembleConfig()
        self.device = self.config.device
        self.scaler = StandardScaler()
        
        # Results storage
        self.results = {}
        self.timing_results = {}
        
    def prepare_data(self, X, y, train_ratio=0.75):
        """Prepare data for GPU training."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        split_idx = int(train_ratio * len(X_scaled))
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X_train_gpu': X_train_tensor, 'X_test_gpu': X_test_tensor,
            'y_train_gpu': y_train_tensor, 'y_test_gpu': y_test_tensor
        }
    
    def train_cpu_models(self, data):
        """Train CPU-based models (LGB, XGB, RF) for ensemble."""
        print("🔄 Training CPU models...")
        
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        n_targets = y_train.shape[1]
        
        models = {'lgb': [], 'xgb': [], 'rf': []}
        predictions = {'lgb': [], 'xgb': [], 'rf': []}
        
        start_time = time.time()
        
        for i in range(n_targets):
            # LightGBM (CPU only due to OpenCL compatibility)
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_seed,
                verbose=-1,
                device='cpu'
            )
            lgb_model.fit(X_train, y_train[:, i])
            models['lgb'].append(lgb_model)
            predictions['lgb'].append(lgb_model.predict(X_test))
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_seed,
                verbosity=0,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
            )
            xgb_model.fit(X_train, y_train[:, i])
            models['xgb'].append(xgb_model)
            predictions['xgb'].append(xgb_model.predict(X_test))
            
            # Random Forest (CPU only)
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=self.config.random_seed,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train[:, i])
            models['rf'].append(rf_model)
            predictions['rf'].append(rf_model.predict(X_test))
        
        training_time = time.time() - start_time
        self.timing_results['cpu_models'] = training_time
        
        # Stack predictions
        for model_type in predictions:
            predictions[model_type] = np.column_stack(predictions[model_type])
        
        print(f"✅ CPU models trained in {training_time:.2f}s")
        return models, predictions
    
    def train_gpu_neural_ensemble(self, data):
        """Train GPU-accelerated neural ensemble."""
        print("🚀 Training GPU neural ensemble...")
        
        X_train_gpu = data['X_train_gpu']
        y_train_gpu = data['y_train_gpu']
        X_test_gpu = data['X_test_gpu']
        y_test_gpu = data['y_test_gpu']
        
        input_dim = X_train_gpu.shape[1]
        n_targets = y_train_gpu.shape[1]
        
        # Create model
        model = GPUNeuralEnsemble(input_dim, n_targets).to(self.device)
        criterion = SharpeOptimizedLoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Training data loader
        train_dataset = TensorDataset(X_train_gpu, y_train_gpu)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
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
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        self.timing_results['gpu_neural'] = training_time
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            neural_predictions_tensor = model(X_test_gpu).cpu()
            neural_predictions = neural_predictions_tensor.detach().numpy()
        
        print(f"✅ GPU neural ensemble trained in {training_time:.2f}s")
        return model, neural_predictions
    
    def optimize_ensemble_weights(self, predictions_dict, y_test):
        """GPU-accelerated ensemble weight optimization."""
        print("⚡ Optimizing ensemble weights on GPU...")
        
        # Convert predictions to GPU tensors
        prediction_tensors = []
        for model_type, preds in predictions_dict.items():
            prediction_tensors.append(torch.FloatTensor(preds).to(self.device))
        
        y_test_gpu = torch.FloatTensor(y_test).to(self.device)
        n_models = len(prediction_tensors)
        n_targets = y_test.shape[1]
        
        # Weight optimizer
        weight_optimizer = GPUEnsembleWeightOptimizer(n_models, n_targets).to(self.device)
        criterion = SharpeOptimizedLoss().to(self.device)
        optimizer = optim.Adam(weight_optimizer.parameters(), lr=self.config.ensemble_lr)
        
        start_time = time.time()
        
        for epoch in range(self.config.ensemble_epochs):
            optimizer.zero_grad()
            
            weights = weight_optimizer()
            
            # Compute weighted ensemble predictions
            ensemble_pred = torch.zeros_like(y_test_gpu)
            for t in range(n_targets):
                for m, pred_tensor in enumerate(prediction_tensors):
                    ensemble_pred[:, t] += weights[t, m] * pred_tensor[:, t]
            
            loss = criterion(ensemble_pred, y_test_gpu)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Weight optimization epoch {epoch}: Loss = {loss.item():.6f}")
        
        # Get final weights and predictions
        final_weights = weight_optimizer().detach().cpu().numpy()
        
        with torch.no_grad():
            final_ensemble_pred = torch.zeros_like(y_test_gpu)
            for t in range(n_targets):
                for m, pred_tensor in enumerate(prediction_tensors):
                    final_ensemble_pred[:, t] += final_weights[t, m] * pred_tensor[:, t]
        
        optimization_time = time.time() - start_time
        self.timing_results['weight_optimization'] = optimization_time
        
        print(f"✅ Weights optimized in {optimization_time:.2f}s")
        return final_weights, final_ensemble_pred.cpu().detach().numpy()
    
    def calculate_sharpe_score(self, y_true, y_pred):
        """Calculate Sharpe-like competition metric."""
        correlations = []
        for i in range(y_true.shape[1]):
            corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        if len(correlations) == 0:
            return 0.0, 0.0, 0.0
        
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        sharpe_score = mean_corr / std_corr if std_corr > 0 else mean_corr
        
        return mean_corr, std_corr, sharpe_score
    
    def run_comprehensive_experiment(self, X, y, experiment_name="GPU_Ensemble"):
        """Run comprehensive GPU-accelerated ensemble experiment."""
        print(f"\n🚀 Starting {experiment_name} Experiment")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # Prepare data
        data = self.prepare_data(X, y)
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
        
        # Train CPU models
        cpu_models, cpu_predictions = self.train_cpu_models(data)
        
        # Train GPU neural ensemble
        gpu_model, neural_predictions = self.train_gpu_neural_ensemble(data)
        
        # Add neural predictions to ensemble
        all_predictions = {**cpu_predictions, 'neural': neural_predictions}
        
        # Optimize ensemble weights
        optimal_weights, final_predictions = self.optimize_ensemble_weights(
            all_predictions, data['y_test']
        )
        
        # Calculate results for each method
        y_test = data['y_test']
        results = {}
        
        # Individual model results
        for model_type, preds in all_predictions.items():
            mean_corr, std_corr, sharpe = self.calculate_sharpe_score(y_test, preds)
            results[model_type] = {
                'mean_spearman': mean_corr,
                'std_spearman': std_corr,
                'sharpe_score': sharpe
            }
        
        # Optimized ensemble result
        mean_corr, std_corr, sharpe = self.calculate_sharpe_score(y_test, final_predictions)
        results['optimized_ensemble'] = {
            'mean_spearman': mean_corr,
            'std_spearman': std_corr,
            'sharpe_score': sharpe
        }
        
        total_time = time.time() - total_start_time
        self.timing_results['total'] = total_time
        
        # Print results
        print(f"\n📊 {experiment_name} Results")
        print("=" * 60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_score'], reverse=True)
        for i, (method, metrics) in enumerate(sorted_results):
            print(f"{i+1}. {method.upper()}")
            print(f"   Mean Spearman: {metrics['mean_spearman']:.4f}")
            print(f"   Std Spearman: {metrics['std_spearman']:.4f}")
            print(f"   Sharpe Score: {metrics['sharpe_score']:.4f}")
            print()
        
        print(f"⏱️ Timing Results:")
        for phase, duration in self.timing_results.items():
            print(f"   {phase}: {duration:.2f}s")
        print(f"   TOTAL: {total_time:.2f}s")
        
        # Calculate improvement
        best_score = sorted_results[0][1]['sharpe_score']
        baseline_score = results.get('lgb', results.get('xgb', {'sharpe_score': 0}))['sharpe_score']
        if baseline_score > 0:
            improvement = ((best_score - baseline_score) / baseline_score) * 100
            print(f"\n🎯 Best Method: {sorted_results[0][0].upper()}")
            print(f"📈 Improvement over baseline: {improvement:.1f}%")
        
        self.results[experiment_name] = {
            'metrics': results,
            'timing': self.timing_results.copy(),
            'optimal_weights': optimal_weights,
            'best_method': sorted_results[0][0],
            'best_score': best_score
        }
        
        return results

def main():
    """Main execution function for GPU ensemble experiments."""
    print("🚀 GPU-Accelerated Ensemble Experiments for Mitsui Challenge")
    print("=" * 80)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️ GPU not available, falling back to CPU")
    else:
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
    
    # Load competition data
    try:
        print("\n📁 Loading competition data...")
        train_data = pd.read_csv('input/train.csv').head(200)  # Start with smaller dataset
        label_data = pd.read_csv('input/train_labels.csv').head(200)
        
        # Merge data
        merged_data = train_data.merge(label_data, on='date_id', how='inner')
        
        # Prepare features and targets
        feature_columns = [col for col in train_data.columns if col != 'date_id'][:20]  # 20 features
        target_columns = [col for col in label_data.columns if col.startswith('target_')][:10]  # 10 targets
        
        X = merged_data[feature_columns].fillna(0).values
        y = merged_data[target_columns].fillna(0).values
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
        
        # Run experiments
        runner = GPUEnsembleExperimentRunner()
        results = runner.run_comprehensive_experiment(X, y, "GPU_Accelerated_Ensemble")
        
        # Save results
        results_df = pd.DataFrame([
            {
                'method': method,
                'mean_spearman': metrics['mean_spearman'],
                'std_spearman': metrics['std_spearman'],
                'sharpe_score': metrics['sharpe_score']
            }
            for method, metrics in results.items()
        ])
        
        results_df.to_csv('GPU_ENSEMBLE_RESULTS.csv', index=False)
        print(f"\n💾 Results saved to GPU_ENSEMBLE_RESULTS.csv")
        
        print("\n🎉 GPU Ensemble Experiments Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()