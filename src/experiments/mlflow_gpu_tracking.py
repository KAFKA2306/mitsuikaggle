#!/usr/bin/env python3
"""
MLflow GPU Tracking System for Mitsui Challenge
Comprehensive experiment tracking with GPU monitoring and performance analysis
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
import os
import threading
from datetime import datetime
import psutil
import warnings
warnings.filterwarnings('ignore')

class GPUMonitor:
    """Real-time GPU monitoring for MLflow experiments."""
    
    def __init__(self, interval=5):
        self.interval = interval
        self.monitoring = False
        self.gpu_data = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        if torch.cuda.is_available():
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print(f"🔍 GPU monitoring started (interval: {self.interval}s)")
        else:
            print("⚠️ No GPU available for monitoring")
    
    def stop_monitoring(self):
        """Stop GPU monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.gpu_data
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            timestamp = datetime.now().isoformat()
            
            # GPU metrics
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                
                # CPU and system metrics
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                
                data_point = {
                    'timestamp': timestamp,
                    'gpu_memory_used_gb': gpu_memory_used,
                    'gpu_memory_cached_gb': gpu_memory_cached,
                    'gpu_memory_total_gb': gpu_memory_total,
                    'gpu_utilization_percent': gpu_utilization,
                    'cpu_percent': cpu_percent,
                    'system_memory_percent': memory_info.percent,
                    'system_memory_used_gb': memory_info.used / (1024**3)
                }
                
                self.gpu_data.append(data_point)
            
            time.sleep(self.interval)

class MLflowExperimentTracker:
    """MLflow experiment tracking with GPU monitoring."""
    
    def __init__(self, experiment_name="Mitsui_GPU_Experiments", tracking_uri="./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                print(f"📊 Created new MLflow experiment: {experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                print(f"📊 Using existing MLflow experiment: {experiment_name}")
        except Exception as e:
            print(f"⚠️ MLflow setup warning: {e}")
            self.experiment_id = "0"  # Default experiment
        
        # GPU monitor
        self.gpu_monitor = GPUMonitor()
        
        print(f"✅ MLflow tracking initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Tracking URI: {tracking_uri}")
        print(f"   Experiment ID: {self.experiment_id}")
    
    def start_run(self, run_name=None, tags=None):
        """Start a new MLflow run with GPU monitoring."""
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Set tags
        if tags:
            mlflow.set_tags(tags)
        
        # Set default tags
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("gpu_available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            mlflow.set_tag("gpu_name", torch.cuda.get_device_name(0))
            mlflow.set_tag("cuda_version", torch.version.cuda)
        mlflow.set_tag("pytorch_version", torch.__version__)
        
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring()
        
        print(f"🚀 Started MLflow run: {mlflow.active_run().info.run_id}")
    
    def log_model_architecture(self, model, input_shape=None):
        """Log model architecture and parameters."""
        # Model summary
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # Assuming float32
        
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("model_size_mb", round(model_size_mb, 2))
        
        # Architecture details
        if hasattr(model, 'architecture'):
            mlflow.log_param("architecture", json.dumps(model.architecture))
        
        # Layer information
        layer_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
                })
        
        # Save layer info as artifact
        layer_df = pd.DataFrame(layer_info)
        layer_df.to_csv("model_layers.csv", index=False)
        mlflow.log_artifact("model_layers.csv")
        os.remove("model_layers.csv")
        
        print(f"📋 Logged model architecture: {total_params:,} parameters, {model_size_mb:.2f} MB")
    
    def log_training_metrics(self, epoch, train_loss, val_loss=None, metrics=None):
        """Log training metrics for each epoch."""
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        if metrics:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)
    
    def log_competition_metrics(self, sharpe_score, mean_correlation, std_correlation, 
                              individual_correlations=None):
        """Log competition-specific metrics."""
        mlflow.log_metric("sharpe_score", sharpe_score)
        mlflow.log_metric("mean_correlation", mean_correlation)
        mlflow.log_metric("std_correlation", std_correlation)
        
        if individual_correlations:
            for i, corr in enumerate(individual_correlations):
                mlflow.log_metric(f"target_{i}_correlation", corr)
            
            # Additional statistics
            mlflow.log_metric("min_correlation", min(individual_correlations))
            mlflow.log_metric("max_correlation", max(individual_correlations))
            mlflow.log_metric("correlation_range", max(individual_correlations) - min(individual_correlations))
        
        print(f"🎯 Logged competition metrics - Sharpe: {sharpe_score:.4f}")
    
    def log_gpu_summary(self):
        """Log GPU utilization summary."""
        gpu_data = self.gpu_monitor.stop_monitoring()
        
        if gpu_data:
            # Calculate summary statistics
            gpu_memory_used = [d['gpu_memory_used_gb'] for d in gpu_data]
            gpu_utilization = [d['gpu_utilization_percent'] for d in gpu_data]
            cpu_percent = [d['cpu_percent'] for d in gpu_data]
            
            # Log summary metrics
            mlflow.log_metric("avg_gpu_memory_used_gb", np.mean(gpu_memory_used))
            mlflow.log_metric("max_gpu_memory_used_gb", np.max(gpu_memory_used))
            mlflow.log_metric("avg_gpu_utilization_percent", np.mean(gpu_utilization))
            mlflow.log_metric("max_gpu_utilization_percent", np.max(gpu_utilization))
            mlflow.log_metric("avg_cpu_percent", np.mean(cpu_percent))
            
            # Save detailed GPU data
            gpu_df = pd.DataFrame(gpu_data)
            gpu_df.to_csv("gpu_monitoring_data.csv", index=False)
            mlflow.log_artifact("gpu_monitoring_data.csv")
            os.remove("gpu_monitoring_data.csv")
            
            print(f"⚡ Logged GPU monitoring data: {len(gpu_data)} data points")
            print(f"   Avg GPU Memory: {np.mean(gpu_memory_used):.2f} GB")
            print(f"   Max GPU Memory: {np.max(gpu_memory_used):.2f} GB")
            print(f"   Avg GPU Utilization: {np.mean(gpu_utilization):.1f}%")
    
    def log_model(self, model, model_name="pytorch_model"):
        """Log PyTorch model to MLflow."""
        try:
            mlflow.pytorch.log_model(model, model_name)
            print(f"💾 Logged PyTorch model: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to log model: {e}")
    
    def end_run(self):
        """End MLflow run with final GPU summary."""
        self.log_gpu_summary()
        mlflow.end_run()
        print(f"✅ MLflow run completed")

# Example models for demonstration
class SimpleCompetitionModel(nn.Module):
    """Simple model for competition with tracking."""
    
    def __init__(self, input_dim, n_targets, hidden_dims=[128, 64]):
        super().__init__()
        
        self.architecture = {
            'input_dim': input_dim,
            'n_targets': n_targets,
            'hidden_dims': hidden_dims,
            'model_type': 'SimpleCompetitionModel'
        }
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_targets))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_with_mlflow_tracking(model, train_loader, val_loader, device, 
                              loss_function, tracker, epochs=20, lr=0.001):
    """Train model with comprehensive MLflow tracking."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # Log training configuration
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("scheduler", "ReduceLROnPlateau")
    mlflow.log_param("batch_size", train_loader.batch_size)
    mlflow.log_param("train_samples", len(train_loader.dataset))
    mlflow.log_param("val_samples", len(val_loader.dataset))
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
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
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        tracker.log_training_metrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            metrics={
                'epoch_time_seconds': epoch_time,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
        )
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, Time={epoch_time:.2f}s")
    
    total_training_time = time.time() - training_start_time
    mlflow.log_metric("total_training_time_seconds", total_training_time)
    
    print(f"🏁 Training completed in {total_training_time:.2f}s")
    return model

def run_mlflow_experiment_demo():
    """Demonstrate MLflow tracking with GPU monitoring."""
    
    print("🎯 MLflow GPU Tracking Demo for Mitsui Challenge")
    print("=" * 70)
    
    # Initialize tracker
    tracker = MLflowExperimentTracker("Mitsui_GPU_Demo")
    
    # Load sample data
    try:
        train_data = pd.read_csv('input/train.csv').head(150)
        label_data = pd.read_csv('input/train_labels.csv').head(150)
        merged = train_data.merge(label_data, on='date_id', how='inner')
        
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:10]
        target_cols = [col for col in label_data.columns if col.startswith('target_')][:5]
        
        X = merged[feature_cols].fillna(0).values
        y = merged[target_cols].fillna(0).values
        
        print(f"✅ Data loaded: {X.shape}")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Prepare data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run multiple experiments
    experiments = [
        {'hidden_dims': [64, 32], 'lr': 0.001, 'name': 'Small_Model_001'},
        {'hidden_dims': [128, 64], 'lr': 0.01, 'name': 'Medium_Model_01'},
        {'hidden_dims': [256, 128, 64], 'lr': 0.001, 'name': 'Large_Model_001'}
    ]
    
    for exp_config in experiments:
        print(f"\n🧪 Running experiment: {exp_config['name']}")
        print("-" * 50)
        
        # Start MLflow run
        tracker.start_run(
            run_name=exp_config['name'],
            tags={
                'model_type': 'SimpleCompetitionModel',
                'experiment_type': 'demo',
                'hidden_dims': str(exp_config['hidden_dims']),
                'learning_rate': str(exp_config['lr'])
            }
        )
        
        # Create model
        model = SimpleCompetitionModel(
            input_dim=X_train.shape[1],
            n_targets=y_train.shape[1],
            hidden_dims=exp_config['hidden_dims']
        ).to(device)
        
        # Log model architecture
        tracker.log_model_architecture(model)
        
        # Train model
        loss_fn = nn.MSELoss()
        trained_model = train_with_mlflow_tracking(
            model, train_loader, val_loader, device, loss_fn, tracker,
            epochs=15, lr=exp_config['lr']
        )
        
        # Evaluate and log competition metrics
        trained_model.eval()
        with torch.no_grad():
            val_predictions = trained_model(torch.FloatTensor(X_val).to(device))
            val_targets = torch.FloatTensor(y_val).to(device)
        
        # Calculate competition metrics manually
        def manual_correlation(x, y):
            x_vals = x.cpu().tolist() if torch.is_tensor(x) else list(x)
            y_vals = y.cpu().tolist() if torch.is_tensor(y) else list(y)
            
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
        for i in range(y_val.shape[1]):
            corr = manual_correlation(val_predictions[:, i], val_targets[:, i])
            correlations.append(corr)
        
        mean_corr = sum(correlations) / len(correlations)
        std_corr = (sum((c - mean_corr)**2 for c in correlations) / len(correlations))**0.5
        sharpe_score = mean_corr / std_corr if std_corr > 0 else mean_corr
        
        # Log competition metrics
        tracker.log_competition_metrics(
            sharpe_score=sharpe_score,
            mean_correlation=mean_corr,
            std_correlation=std_corr,
            individual_correlations=correlations
        )
        
        # Log model
        tracker.log_model(trained_model, f"model_{exp_config['name']}")
        
        # End run
        tracker.end_run()
        
        print(f"✅ Experiment {exp_config['name']} completed")
        print(f"   Sharpe Score: {sharpe_score:.4f}")
        print(f"   Mean Correlation: {mean_corr:.4f}")
    
    print(f"\n🎉 All MLflow experiments completed!")
    print(f"📊 View results at: {tracker.tracking_uri}")
    print(f"💡 Run: mlflow ui --backend-store-uri {tracker.tracking_uri}")

def main():
    """Main execution for MLflow tracking demo."""
    print("📊 MLflow GPU Tracking System for Mitsui Challenge")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️ Warning: GPU not available, using CPU")
    else:
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
    
    print(f"✅ MLflow Version: {mlflow.__version__}")
    
    # Run demonstration
    run_mlflow_experiment_demo()

if __name__ == "__main__":
    main()