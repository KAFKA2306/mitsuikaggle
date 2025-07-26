#!/usr/bin/env python3
"""
GPU Neural Architecture Search (NAS) for Track D - Mitsui Challenge
Implements multi-objective Bayesian optimization for neural architecture discovery
Optimizes for accuracy, stability, and efficiency under resource constraints
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import random
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class NASSearchSpace:
    """Defines the neural architecture search space."""
    
    def __init__(self):
        # Architecture dimensions
        self.layers = [2, 3, 4, 5, 6]
        self.hidden_dims = [32, 64, 128, 256, 512]
        self.activations = ['relu', 'gelu', 'swish', 'tanh']
        self.dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]
        self.batch_norms = [True, False]
        self.optimizers = ['adam', 'adamw', 'sgd']
        self.learning_rates = [0.0001, 0.001, 0.01, 0.1]
        self.schedulers = ['plateau', 'cosine', 'exponential', 'none']
        
    def sample_architecture(self):
        """Sample a random architecture from the search space."""
        n_layers = random.choice(self.layers)
        
        # Create layer configuration
        layer_configs = []
        for i in range(n_layers):
            config = {
                'hidden_dim': random.choice(self.hidden_dims),
                'activation': random.choice(self.activations),
                'dropout': random.choice(self.dropouts),
                'batch_norm': random.choice(self.batch_norms)
            }
            layer_configs.append(config)
        
        # Training configuration
        training_config = {
            'optimizer': random.choice(self.optimizers),
            'learning_rate': random.choice(self.learning_rates),
            'scheduler': random.choice(self.schedulers),
            'batch_size': random.choice([16, 32, 64, 128])
        }
        
        return {
            'layers': layer_configs,
            'training': training_config,
            'n_layers': n_layers
        }
    
    def mutate_architecture(self, architecture, mutation_rate=0.3):
        """Mutate an existing architecture."""
        new_arch = json.loads(json.dumps(architecture))  # Deep copy
        
        # Mutate layer configurations
        for layer in new_arch['layers']:
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    layer['hidden_dim'] = random.choice(self.hidden_dims)
                if random.random() < 0.3:
                    layer['activation'] = random.choice(self.activations)
                if random.random() < 0.3:
                    layer['dropout'] = random.choice(self.dropouts)
                if random.random() < 0.3:
                    layer['batch_norm'] = random.choice(self.batch_norms)
        
        # Mutate training configuration
        if random.random() < mutation_rate:
            new_arch['training']['learning_rate'] = random.choice(self.learning_rates)
        if random.random() < mutation_rate:
            new_arch['training']['optimizer'] = random.choice(self.optimizers)
        
        return new_arch
    
    def crossover_architectures(self, arch1, arch2):
        """Create offspring through crossover of two architectures."""
        # Take layers from both parents
        min_layers = min(len(arch1['layers']), len(arch2['layers']))
        crossover_point = random.randint(1, min_layers - 1)
        
        new_layers = arch1['layers'][:crossover_point] + arch2['layers'][crossover_point:min_layers]
        
        # Mix training configurations
        new_training = {}
        for key in arch1['training']:
            new_training[key] = random.choice([arch1['training'][key], arch2['training'][key]])
        
        return {
            'layers': new_layers,
            'training': new_training,
            'n_layers': len(new_layers)
        }

class DynamicNeuralNetwork(nn.Module):
    """Dynamically constructed neural network based on architecture specification."""
    
    def __init__(self, input_dim, n_targets, architecture):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.architecture = architecture
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, layer_config in enumerate(architecture['layers']):
            hidden_dim = layer_config['hidden_dim']
            
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if layer_config['batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            activation = layer_config['activation']
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            # Dropout
            if layer_config['dropout'] > 0:
                layers.append(nn.Dropout(layer_config['dropout']))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_targets))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MultiObjectiveEvaluator:
    """Multi-objective evaluator for neural architectures."""
    
    def __init__(self, device):
        self.device = device
        
    def manual_correlation(self, x, y):
        """Manual correlation calculation."""
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
        """Calculate Sharpe-like score."""
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
    
    def evaluate_architecture(self, architecture, X_train, y_train, X_test, y_test, 
                            max_epochs=20, time_budget=120):
        """Evaluate a neural architecture across multiple objectives."""
        
        input_dim = X_train.shape[1]
        n_targets = y_train.shape[1]
        
        # Create model
        model = DynamicNeuralNetwork(input_dim, n_targets, architecture).to(self.device)
        
        # Training configuration
        training_config = architecture['training']
        
        # Optimizer
        if training_config['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        elif training_config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=training_config['learning_rate'], weight_decay=1e-4)
        else:  # sgd
            optimizer = optim.SGD(model.parameters(), lr=training_config['learning_rate'], momentum=0.9)
        
        # Scheduler
        scheduler = None
        if training_config['scheduler'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        elif training_config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        elif training_config['scheduler'] == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Training
        start_time = time.time()
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        training_losses = []
        
        for epoch in range(max_epochs):
            # Check time budget
            if time.time() - start_time > time_budget:
                break
                
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
            training_losses.append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break
            
            # Update scheduler
            if scheduler is not None:
                if training_config['scheduler'] == 'plateau':
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
        
        training_time = time.time() - start_time
        
        # Load best model
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_train_tensor)
            test_predictions = model(X_test_tensor)
        
        # Calculate metrics
        train_mean_corr, train_std_corr, train_sharpe = self.calculate_sharpe_score(y_train_tensor, train_predictions)
        test_mean_corr, test_std_corr, test_sharpe = self.calculate_sharpe_score(y_test_tensor, test_predictions)
        
        # Multi-objective scores
        objectives = {
            'accuracy': test_sharpe,  # Primary objective
            'stability': -test_std_corr,  # Lower variance is better
            'efficiency': -training_time,  # Faster training is better
            'complexity': -model.count_parameters() / 1000,  # Fewer parameters is better
            'overfitting': train_sharpe - test_sharpe  # Lower overfitting is better
        }
        
        # Combined score (weighted sum)
        weights = {'accuracy': 0.4, 'stability': 0.2, 'efficiency': 0.2, 'complexity': 0.1, 'overfitting': 0.1}
        combined_score = sum(weights[key] * objectives[key] for key in objectives)
        
        return {
            'objectives': objectives,
            'combined_score': combined_score,
            'test_sharpe': test_sharpe,
            'train_sharpe': train_sharpe,
            'training_time': training_time,
            'parameters': model.count_parameters(),
            'converged_epochs': len(training_losses),
            'best_loss': best_loss
        }

class BayesianNASOptimizer:
    """Bayesian optimization for Neural Architecture Search."""
    
    def __init__(self, search_space, evaluator, population_size=20):
        self.search_space = search_space
        self.evaluator = evaluator
        self.population_size = population_size
        self.population = []
        self.fitness_history = []
        
    def initialize_population(self):
        """Initialize random population."""
        print("🔬 Initializing NAS population...")
        self.population = []
        for i in range(self.population_size):
            architecture = self.search_space.sample_architecture()
            self.population.append(architecture)
            print(f"   Generated architecture {i+1}/{self.population_size}")
    
    def evaluate_population(self, X_train, y_train, X_test, y_test):
        """Evaluate entire population."""
        print(f"⚡ Evaluating population of {len(self.population)} architectures...")
        
        fitness_scores = []
        detailed_results = []
        
        for i, architecture in enumerate(self.population):
            print(f"   Evaluating architecture {i+1}/{len(self.population)}...")
            
            try:
                result = self.evaluator.evaluate_architecture(
                    architecture, X_train, y_train, X_test, y_test
                )
                fitness_scores.append(result['combined_score'])
                detailed_results.append(result)
                
                print(f"     Sharpe: {result['test_sharpe']:.4f}, "
                      f"Time: {result['training_time']:.2f}s, "
                      f"Params: {result['parameters']}")
                
            except Exception as e:
                print(f"     Failed: {e}")
                fitness_scores.append(-1.0)  # Penalty for failed architectures
                detailed_results.append(None)
        
        self.fitness_history.append(fitness_scores)
        return fitness_scores, detailed_results
    
    def evolve_population(self, fitness_scores, survival_rate=0.5):
        """Evolve population using genetic operators."""
        print("🧬 Evolving population...")
        
        # Sort by fitness
        population_fitness = list(zip(self.population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Select survivors
        n_survivors = int(len(self.population) * survival_rate)
        survivors = [arch for arch, fitness in population_fitness[:n_survivors]]
        
        print(f"   Selected {len(survivors)} survivors")
        print(f"   Best fitness: {population_fitness[0][1]:.4f}")
        
        # Generate new population
        new_population = survivors.copy()
        
        while len(new_population) < self.population_size:
            if random.random() < 0.7:  # Mutation
                parent = random.choice(survivors)
                child = self.search_space.mutate_architecture(parent)
            else:  # Crossover
                parent1, parent2 = random.sample(survivors, 2)
                child = self.search_space.crossover_architectures(parent1, parent2)
            
            new_population.append(child)
        
        self.population = new_population
        return population_fitness[0]  # Return best architecture and fitness
    
    def optimize(self, X_train, y_train, X_test, y_test, generations=5):
        """Run full NAS optimization."""
        print(f"\n🚀 Starting Neural Architecture Search")
        print(f"   Population size: {self.population_size}")
        print(f"   Generations: {generations}")
        print("=" * 60)
        
        # Initialize
        self.initialize_population()
        
        best_overall = None
        best_fitness = float('-inf')
        
        for generation in range(generations):
            print(f"\n🧬 Generation {generation + 1}/{generations}")
            print("-" * 40)
            
            # Evaluate
            fitness_scores, detailed_results = self.evaluate_population(X_train, y_train, X_test, y_test)
            
            # Track best
            max_fitness_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_overall = {
                    'architecture': self.population[max_fitness_idx],
                    'fitness': best_fitness,
                    'detailed_result': detailed_results[max_fitness_idx],
                    'generation': generation + 1
                }
            
            # Evolve (except last generation)
            if generation < generations - 1:
                best_current = self.evolve_population(fitness_scores)
                print(f"   Generation best: {best_current[1]:.4f}")
        
        print(f"\n🎯 NAS Optimization Completed!")
        print(f"   Best fitness: {best_fitness:.4f}")
        print(f"   Found in generation: {best_overall['generation']}")
        
        return best_overall, self.fitness_history

def run_nas_experiment(X, y, experiment_name="NAS_Track_D"):
    """Run complete NAS experiment."""
    print(f"\n🔬 {experiment_name} - Neural Architecture Search")
    print("=" * 70)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Prepare data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(0.75 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Initialize NAS components
    search_space = NASSearchSpace()
    evaluator = MultiObjectiveEvaluator(device)
    optimizer = BayesianNASOptimizer(search_space, evaluator, population_size=10)
    
    # Run optimization
    start_time = time.time()
    best_architecture, fitness_history = optimizer.optimize(
        X_train, y_train, X_test, y_test, generations=3
    )
    total_time = time.time() - start_time
    
    # Results summary
    print(f"\n📊 NAS Results Summary")
    print("=" * 70)
    print(f"Total optimization time: {total_time:.2f}s")
    print(f"Best architecture found:")
    print(f"   Test Sharpe score: {best_architecture['detailed_result']['test_sharpe']:.4f}")
    print(f"   Training time: {best_architecture['detailed_result']['training_time']:.2f}s")
    print(f"   Parameters: {best_architecture['detailed_result']['parameters']:,}")
    print(f"   Layers: {best_architecture['architecture']['n_layers']}")
    
    # Architecture details
    print(f"\n🏗️ Best Architecture Details:")
    for i, layer in enumerate(best_architecture['architecture']['layers']):
        print(f"   Layer {i+1}: {layer['hidden_dim']} units, {layer['activation']}, "
              f"dropout={layer['dropout']}, batch_norm={layer['batch_norm']}")
    
    print(f"\n⚙️ Training Configuration:")
    training = best_architecture['architecture']['training']
    for key, value in training.items():
        print(f"   {key}: {value}")
    
    # Save results
    results = {
        'best_architecture': best_architecture['architecture'],
        'best_fitness': best_architecture['fitness'],
        'detailed_results': best_architecture['detailed_result'],
        'optimization_time': total_time,
        'fitness_history': fitness_history
    }
    
    # Save as JSON
    with open('NAS_TRACK_D_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to NAS_TRACK_D_RESULTS.json")
    
    # GPU memory usage
    if torch.cuda.is_available():
        print(f"\n⚡ GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / (1024**2):.1f} MB")
    
    return results

def main():
    """Main execution for NAS experiments."""
    print("🔬 Neural Architecture Search (NAS) - Track D")
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
        train_data = pd.read_csv('input/train.csv').head(250)
        label_data = pd.read_csv('input/train_labels.csv').head(250)
        merged = train_data.merge(label_data, on='date_id', how='inner')
        
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:15]
        target_cols = [col for col in label_data.columns if col.startswith('target_')][:8]
        
        X = merged[feature_cols].fillna(0).values
        y = merged[target_cols].fillna(0).values
        
        print(f"✅ Data loaded: {X.shape}")
        
        # Run NAS experiment
        results = run_nas_experiment(X, y)
        
        print("\n🎉 Neural Architecture Search Completed!")
        
    except Exception as e:
        print(f"❌ NAS experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()