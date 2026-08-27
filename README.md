https://kafka2306.github.io/mitsuikaggle/

# MITSUI&CO. Commodity Prediction Challenge research

[![Research artifact checks](https://github.com/KAFKA2306/mitsuikaggle/actions/workflows/docs-simple.yml/badge.svg)](https://github.com/KAFKA2306/mitsuikaggle/actions/workflows/docs-simple.yml)
[![Competition](https://img.shields.io/badge/Kaggle-Competition-orange)](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)

This repository is for reproducible commodity-prediction research around the **MITSUI&CO. Commodity Prediction Challenge**. The official Kaggle competition page remains the authority for competition rules, evaluation, and leaderboard results.

No verified Kaggle submission ID, public score, or private score is currently bound to the repository artifacts below. Local scores must not be presented as leaderboard performance.

## Vision

Make every reported result answer four questions directly: **which data, which split, which metric, and which baseline**. The value of the repository is not the number of models it contains; it is the ability to inspect a claim, reproduce its conditions, and choose the next experiment without mixing local evidence with competition evidence.

## Design philosophy

- Keep leaderboard results, local experiments, and planned work separate.
- Bind reported scores to exact data and code revisions where evidence exists.
- Preserve unknown historical metadata as unknown rather than inferring it.
- Prefer chronological held-out evaluation and explicit leakage boundaries.
- Treat the official competition metric and evaluation API as a separate authority until reproduced exactly.

## Current verified repository evidence

### Local 5-target subset comparison

[`results/experiments/direct_execution_provenance.json`](results/experiments/direct_execution_provenance.json) binds the historical three-model comparison to:

- producer: `scripts/DIRECT_EXECUTION_RESULTS.py` blob `9f405c6b6b202238ecfaa1fb13283854ac3dd8a9`
- input `train.csv` blob `70c3f303d92c99016fb8d3c47adb6d3483ee18a6`
- input `train_labels.csv` blob `a13e620c49e71816d4be4482753a54aa2e31bb84`
- first 200 rows
- 10 features
- 5 targets
- chronological 75/25 split: 150 train / 50 holdout rows
- random seed: 42

[`results/experiments/ACTUAL_EXPERIMENT_RESULTS.csv`](results/experiments/ACTUAL_EXPERIMENT_RESULTS.csv) stores:

| Method | Stored Sharpe-like score |
|---|---:|
| Multi-Model Ensemble (LGB+XGB+RF) | 0.812489 |
| Classical Ensemble (LGB+XGB) | 0.646361 |
| Single Model (LightGBM) | 0.366280 |

The stored multi-model score is **121.8219647% higher** than the stored LightGBM score under this repository-local metric. That arithmetic is verified. It is **not** a verified Kaggle leaderboard improvement.

The local metric computes one Spearman correlation per selected target over the 50 holdout rows, then divides the mean of those five target-level correlations by their standard deviation. Exact equivalence to the competition metric implementation has not been demonstrated.

Historical package versions, exact run timestamp/runtime, direct producer-to-persisted-file proof, and Kaggle submission identity remain unknown.

### Historical local 424-target run

[`results/experiments/production_424_provenance.json`](results/experiments/production_424_provenance.json) binds [`results/experiments/production_424_results.json`](results/experiments/production_424_results.json) to:

- producer: `scripts/final_424_production.py` blob `7fbde6a3731418e73283ed1a3c4b82d564796e07`
- the same two input blobs above
- 1,917 reported merged samples
- 557 features
- 424 targets
- chronological 80/20 split
- stored score: `1.1911680408317646`
- recorded runtime: `15.128787553310394` minutes
- recorded device: `cuda`

This run is useful as evidence that a 424-target local GPU experiment was executed, but **not** as competition-equivalent validation evidence. Its metric uses Pearson correlation rather than Spearman, its holdout is reused for checkpoint selection / early stopping and final reporting, and the producer does not pin a random seed.

## Competition context

The official Kaggle page defines the competition metric as a variant of the Sharpe ratio: the mean Spearman rank correlation between predictions and targets divided by its standard deviation. Submissions use Kaggle's evaluation API to prevent forward-looking access.

Until the repository reproduces the authoritative metric code and its aggregation, tie, missing-value, and standard-deviation semantics exactly, repository-local scores are labeled with their local definitions instead of being called Kaggle scores.

Official competition: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge

## Reproduce or inspect the evidence

The currently reliable entry points are the persisted provenance artifacts:

```text
results/experiments/direct_execution_provenance.json
results/experiments/production_424_provenance.json
results/experiments/competition_metric_reference.json
```

The repository does not yet claim exact historical re-execution because historical dependency/runtime versions are not fully bound. Re-running a script today would be a new experiment unless its environment and output identity are explicitly recorded.

## Executed and planned research

**Executed / preserved**

- 5-target / 200-row local ensemble comparison with reconstructed provenance
- historical 424-target local GPU run with explicit leakage/metric limitations
- audit separating repository-local metrics from the official competition metric

**Next research gate**

- reproduce the authoritative Kaggle metric implementation as a deterministic local contract
- then run a new 424-target benchmark with fixed seed and train / validation / untouched-test separation
- persist exact code, data, environment, predictions, and evaluation output for that new run

Model or feature expansion is lower priority until those evaluation conditions are fixed.

## Research files

- [`results/experiments/`](results/experiments/) — persisted results and provenance
- [`scripts/`](scripts/) — historical experiment producers
- [`src/experiments/`](src/experiments/) — experiment implementations
- [`research/`](research/) — research material
- [`notebooks/`](notebooks/) — notebook work
- [`docs/`](docs/) — supporting documentation
- [`input/`](input/) — competition input data in the repository

Do not infer current competition rules or leaderboard standing from older repository notes; use the official Kaggle competition page.

## Scope

A local improvement, a persisted experiment result, an official competition score, and realized trading performance are different claims. This repository keeps them separate until direct evidence connects them.
