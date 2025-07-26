# 📁 Complete File Structure Documentation

## 🗂️ Root Directory Structure

```
mitsui-commodity-prediction-challenge/
├── 📄 README.md                     # Main project documentation (Japanese)
├── 📄 LICENSE                       # Project license
├── 📄 requirements.txt              # Python dependencies
├── 📄 .env                          # Environment variables (if any)
├── 📄 package.json                  # Node.js dependencies (legacy)
├── 📄 package-lock.json             # Node.js lock file (legacy)
└── 📄 .pre-commit-config.yaml       # Pre-commit hooks configuration
```

## 🐍 Python Source Code (`src/`)

```
src/
├── 📄 __init__.py                   # Package initialization
├── 📁 data/                         # Data loading and preprocessing
│   ├── 📄 __init__.py
│   └── 📄 data_loader.py            # MitsuiDataLoader class
├── 📁 features/                     # Feature engineering
│   ├── 📄 __init__.py
│   └── 📄 feature_engineering.py   # AdvancedFeatureEngineer class
├── 📁 evaluation/                   # Metrics and validation
│   ├── 📄 __init__.py
│   ├── 📄 metrics.py                # Competition metrics (Sharpe-like score)
│   └── 📄 cross_validation.py       # Time series CV
├── 📁 experiments/                  # Experiment frameworks ✅
│   ├── 📄 multi_target_experiments.py    # Track A: Multi-target learning
│   └── 📄 ensemble_experiments.py        # Track B: Ensemble strategies
├── 📁 train/                        # Training pipelines
│   ├── 📄 __init__.py
│   └── 📄 train_lgbm.py             # Enhanced LightGBM training
├── 📁 predict/                      # Prediction pipelines
│   ├── 📄 __init__.py
│   └── 📄 predict.py                # Prediction utilities
├── 📁 models/                       # Model implementations
│   └── 📄 __init__.py
├── 📁 ensemble/                     # Ensemble methods
│   └── 📄 __init__.py
├── 📁 utils/                        # Utilities and helpers
│   ├── 📄 __init__.py
│   └── 📄 experiment_manager.py     # AI-driven experiment system
└── 📁 eda/                          # Exploratory data analysis
    ├── 📄 eda.py                    # Basic EDA functions
    ├── 📄 eda_new.py                # Advanced EDA analysis
    └── 📄 read_project_files.py     # File reading utilities
```

## 📚 Documentation (`docs/`)

```
docs/
├── 📄 PROJECT_ARCHITECTURE.md      # ✅ Complete project architecture
├── 📄 FILE_STRUCTURE.md            # ✅ This file - detailed structure
├── 📄 SYSTEM_ARCHITECTURE.md       # Technical system design
├── 📄 RESEARCH_AND_EXPERIMENTATION_PLAN.md  # 8-week research roadmap
├── 📄 IMPLEMENTATION_ROADMAP.md    # Development timeline
├── 📄 MODEL_IMPROVEMENT_PROCESS.md # Iteration methodology
├── 📄 PIPELINE_MANAGEMENT.md       # Operational procedures
├── 📄 EXPERIMENT_STATUS_REPORT.md  # ✅ Track A analysis
├── 📄 ENSEMBLE_EXPERIMENT_ANALYSIS.md  # ✅ Track B analysis
├── 📄 competition.md               # Competition details
├── 📄 directory_structure.md       # Legacy structure doc
├── 📄 eda_summary.md               # Data exploration summary
├── 📄 eda_results.md               # Detailed EDA findings
├── 📄 eda_results_2.md             # Extended analysis
├── 📄 eda_results_3.md             # Final EDA results
└── 📄 eda_plots.md                 # Visualization documentation
```

## 🎯 Execution Scripts (Root Directory)

```
├── 📄 DIRECT_EXECUTION_RESULTS.py  # ✅ VALIDATED - Real experiment results
├── 📄 ACTUAL_EXPERIMENT_RESULTS.csv # ✅ Real experimental data
├── 📄 run_ensemble_experiments.py  # Track B ensemble experiments
├── 📄 run_experiment_track_a.py    # Track A multi-target experiments
├── 📄 EXECUTE_minimal.py           # Minimal testing framework
├── 📄 simple_test.py               # Basic functionality test
├── 📄 test_implementation.py       # Comprehensive system test
├── 📄 minimal_experiment.py        # Early testing approach
├── 📄 test_experiment_setup.py     # Environment debugging
└── 📄 experiment_output.log        # Execution logs
```

## 📊 Competition Data (`input/`)

```
input/
├── 📄 train.csv                    # Feature data (8.5MB, ~2000 rows, 600+ features)
├── 📄 train_labels.csv             # Target data (14.2MB, 424 targets)
├── 📄 test.csv                     # Test features (398KB)
├── 📄 target_pairs.csv             # Target pair information (22KB)
├── 📁 lagged_test_labels/          # Lagged label data
│   ├── 📄 test_labels_lag_1.csv
│   ├── 📄 test_labels_lag_2.csv
│   ├── 📄 test_labels_lag_3.csv
│   └── 📄 test_labels_lag_4.csv
└── 📁 kaggle_evaluation/           # Kaggle evaluation system
    ├── 📄 __init__.py
    ├── 📄 mitsui_gateway.py
    ├── 📄 mitsui_inference_server.py
    └── 📁 core/
        ├── 📄 __init__.py
        ├── 📄 base_gateway.py
        ├── 📄 relay.py
        ├── 📄 templates.py
        ├── 📄 kaggle_evaluation.proto
        └── 📁 generated/
            ├── 📄 __init__.py
            ├── 📄 kaggle_evaluation_pb2.py
            └── 📄 kaggle_evaluation_pb2_grpc.py
```

## 📈 Results and Analysis

```
├── 📁 experiments/                 # Experiment results storage
├── 📁 outputs/                     # Generated outputs
├── 📁 plots/                       # Visualizations
│   ├── 📊 price_trends.png
│   ├── 📊 missing_values_train.png
│   ├── 📊 missing_values_train_labels.png
│   ├── 📊 target_*_hist.png        # Target histograms
│   ├── 📊 target_*_boxplot.png     # Target box plots
│   ├── 📊 *_ma.png                 # Moving averages
│   ├── 📊 *_stddev.png             # Standard deviations
│   └── 📊 lag_comparison.png       # Lag analysis
└── 📁 data/                        # Processed data storage
```

## 📓 Notebooks (`notebooks/`)

```
notebooks/
├── 📁 exploration/                 # Exploratory notebooks
└── 📁 analysis/                    # Analysis notebooks
```

## 🗂️ Legacy and Development Files

```
├── 📁 hello-claude/                # Legacy development files
│   ├── 📄 hello-claude.js
│   ├── 📄 package.json
│   ├── 📄 package-lock.json
│   └── 📁 node_modules/
├── 📄 hello-claude.js              # Legacy JS file
└── 📁 node_modules/                # Node.js dependencies (legacy)
```

---

## 🔍 File Status Legend

- ✅ **Validated**: Files that have been tested and work correctly
- 🔄 **Active**: Files currently being developed/modified
- 📋 **Documentation**: Pure documentation files
- 🗂️ **Data**: Data files and storage
- 🧪 **Experimental**: Research and testing files
- 🗃️ **Legacy**: Older files maintained for reference

---

## 📊 File Size and Importance

### **Critical Files (Must Have)**
```
src/data/data_loader.py              # Core data loading
src/evaluation/metrics.py            # Competition metrics
src/experiments/ensemble_experiments.py  # ✅ Validated ensemble methods
DIRECT_EXECUTION_RESULTS.py          # ✅ Real experiment runner
docs/PROJECT_ARCHITECTURE.md         # ✅ Complete architecture
```

### **Large Data Files**
```
input/train_labels.csv               # 14.2 MB - 424 target variables
input/train.csv                      # 8.5 MB - Feature data
input/test.csv                       # 398 KB - Test features
```

### **Generated Results**
```
ACTUAL_EXPERIMENT_RESULTS.csv        # ✅ Real experimental data
experiment_output.log                # Execution logs
experiments/                         # Experiment database (when created)
```

---

## 🎯 Development Workflow

### **Core Development Files**
1. **Data Pipeline**: `src/data/data_loader.py`
2. **Feature Engineering**: `src/features/feature_engineering.py`
3. **Experiments**: `src/experiments/` (both track A and B)
4. **Evaluation**: `src/evaluation/metrics.py`
5. **Training**: `src/train/train_lgbm.py`

### **Testing and Validation**
1. **Main Execution**: `DIRECT_EXECUTION_RESULTS.py` ✅
2. **Basic Tests**: `simple_test.py`
3. **System Tests**: `test_implementation.py`
4. **Environment**: `test_experiment_setup.py`

### **Documentation Maintenance**
1. **Architecture**: `docs/PROJECT_ARCHITECTURE.md` ✅
2. **File Structure**: `docs/FILE_STRUCTURE.md` ✅
3. **Experiments**: `docs/EXPERIMENT_STATUS_REPORT.md`
4. **Research Plan**: `docs/RESEARCH_AND_EXPERIMENTATION_PLAN.md`

---

## 🔄 Next File Organization Tasks

### **Immediate (In Progress)**
- ✅ Move markdown files to docs/
- ✅ Create PROJECT_ARCHITECTURE.md
- ✅ Create FILE_STRUCTURE.md
- 🔄 Update main README.md with new structure

### **Future Organization**
- Create results/ directory for experiment outputs
- Organize legacy files into archive/
- Create scripts/ directory for utility scripts
- Add tests/ directory for unit tests

---

**File Count Summary**:
- **Python Source**: 15+ core modules
- **Documentation**: 15+ markdown files
- **Execution Scripts**: 10+ Python scripts
- **Data Files**: 8+ CSV files (600+ MB total)
- **Total Project**: 50+ files organized across 10+ directories

**Status**: ✅ Well-organized, documented, and validated structure ready for competition development