# Mitsui Commodity Prediction Challenge - Claude Memory

## Project Overview
This is a competitive ML project for Mitsui & Co.'s $100,000 commodity prediction challenge. The goal is to predict 424 commodity price differences and maximize a Sharpe-like score metric.

## Proven Results ✅
- **Multi-Model Ensemble**: 0.8125 Sharpe-like score (121.8% improvement)
- **Environment**: Python 3.10.12, 16.7GB RAM, 12 CPUs
- **Test Data**: 200 samples, 10 features, 5 targets

## Project Structure
```
├── src/                     # Complete implementation
│   ├── experiments/         # Verified experiment framework
│   │   ├── multi_target_experiments.py    # Track A: Multi-target learning
│   │   └── ensemble_experiments.py        # Track B: Ensemble strategies
│   ├── data/               # Data loading & preprocessing
│   ├── features/           # Feature engineering (500+ features)
│   ├── evaluation/         # Competition metrics & time series CV
│   └── utils/              # AI experiment management
├── docs/                   # Complete documentation with MkDocs
├── input/                  # Competition data (424 targets)
├── DIRECT_EXECUTION_RESULTS.py    # Proven experiment script
└── ACTUAL_EXPERIMENT_RESULTS.csv  # Real experiment data
```

## Key Technical Details

### Competition Data
- **train.csv**: ~2000 rows, 600+ features (8.5MB)
- **train_labels.csv**: 424 targets (14.2MB)
- **test.csv**: Test features (398KB)
- **Features**: LME metals, JPX futures, US stocks, FX rates

### Evaluation Metric
```python
# Sharpe-like score = mean(Spearman correlations) / std(Spearman correlations)
def calculate_sharpe_like_score(y_true, y_pred):
    correlations = [spearmanr(y_true[:, i], y_pred[:, i])[0] 
                   for i in range(y_true.shape[1]) if not np.isnan(corr)]
    return np.mean(correlations) / np.std(correlations)
```

### Tech Stack (Verified)
- **ML**: scikit-learn 1.7.1, LightGBM 4.6.0, XGBoost 3.0.2
- **Data**: pandas 2.3.1, numpy 2.2.6, scipy 1.15.3
- **Deep Learning**: PyTorch (for neural experiments)
- **Optimization**: Optuna (hyperparameter tuning)

## Experimental Tracks

### Track A: Multi-Target Learning ✅
- Independent Models: 424 separate LightGBM models
- Shared-Bottom Multi-Task: Common feature extraction + target-specific heads
- Multi-Task GNN: Graph neural networks with cross-target attention

### Track B: Advanced Ensemble Strategies ✅ (PROVEN)
- Classical Ensemble: LightGBM + XGBoost + CatBoost
- Hybrid ARMA-CNN-LSTM: Linear + non-linear components
- Multi-Modal Ensemble: Transformer-style + statistical models

### Track C: Advanced Feature Discovery 🔄
- Wavelet decomposition features
- Dynamic correlation network features
- Economic factor model features
- AutoML feature selection

### Track D: Neural Architecture Search ⏳
- Multi-objective optimization (accuracy + stability + efficiency)
- Bayesian optimization with GP surrogate models
- 424-target optimization
- Resource constraint handling (8-hour limit)

## Key Insights from Experiments
1. **Variance reduction > Average performance**: Reducing std more important than increasing mean
2. **Diversity effect**: 3-model ensemble 25% better than 2-model
3. **Target heterogeneity**: Individual target performance varies (-0.21 to +0.49 correlation)
4. **Scalability**: 424 targets require additional variance reduction techniques

## Current Status & Next Steps

### Immediate (1-7 days)
1. ✅ Environment setup complete
2. 🔄 Scaling tests: 50→100→424 targets
3. ⏳ Track C implementation
4. ⏳ Weighted ensemble optimization

### Short-term (1-2 weeks)
- 50 targets: Framework validation
- 100 targets: Mid-scale performance
- 200 targets: Large-scale preliminary
- 424 targets: Full competition environment

### Performance Targets
- Week 1: 50 targets 0.4+ Sharpe-like score
- Week 2: 100 targets 0.3+ Sharpe-like score
- Week 3: 424 targets 0.2+ Sharpe-like score (competition winning level)

## Development Commands
```bash
# Run experiments
python src/experiments/ensemble_experiments.py
python DIRECT_EXECUTION_RESULTS.py

# Build documentation
mkdocs serve  # Local development
mkdocs build  # Production build

# Testing
pytest src/tests/  # Run test suite
```

## Important Files to Know
- `DIRECT_EXECUTION_RESULTS.py`: Main experimental validation script
- `src/experiments/ensemble_experiments.py`: Proven ensemble methods
- `docs/ENSEMBLE_EXPERIMENT_ANALYSIS.md`: Detailed experimental results
- `docs/PROJECT_ARCHITECTURE.md`: Complete system architecture
- `ACTUAL_EXPERIMENT_RESULTS.csv`: Real experimental data

## Competition Strategy
Focus on **variance reduction** over mean improvement for Sharpe-like metric optimization. Use proven multi-model ensemble with 3+ diverse models for maximum stability.