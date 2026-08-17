# MITSUI&CO. Commodity Prediction Challenge research

[![Research artifact checks](https://github.com/KAFKA2306/mitsuikaggle/actions/workflows/docs-simple.yml/badge.svg)](https://github.com/KAFKA2306/mitsuikaggle/actions/workflows/docs-simple.yml)
[![Competition](https://img.shields.io/badge/Kaggle-Competition-orange)](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)

This repository contains local research for the **MITSUI&CO. Commodity Prediction Challenge**. The official competition page is the authority for competition rules, evaluation and leaderboard results.

The repository currently contains local experiment artifacts, but no verified leaderboard result is identified in the current tree. Local scores below must therefore not be presented as Kaggle leaderboard performance.

## Current verified repository evidence

### Three-model comparison

[`results/experiments/ACTUAL_EXPERIMENT_RESULTS.csv`](results/experiments/ACTUAL_EXPERIMENT_RESULTS.csv) stores three local results using the columns `mean_spearman`, `std_spearman`, and `sharpe_like_score`:

| Method | Mean Spearman | Std. Spearman | Stored Sharpe-like score |
|---|---:|---:|---:|
| Multi-Model Ensemble (LGB+XGB+RF) | 0.120943 | 0.148854 | 0.812489 |
| Classical Ensemble (LGB+XGB) | 0.130775 | 0.202325 | 0.646361 |
| Single Model (LightGBM) | 0.079544 | 0.217168 | 0.366280 |

The stored multi-model score is about 121.8% higher than the stored single-model score. This is a comparison between rows in this local artifact; it is **not** a verified Kaggle leaderboard improvement.

The CSV does not contain sample count, split definition, seed, dataset hash, execution environment, or the command that generated it. Until those are tied to the artifact, the result is evidence of a stored local comparison rather than a fully reproducible experiment.

### 424-target local artifact

[`results/experiments/production_424_results.json`](results/experiments/production_424_results.json) records another local run with:

- 1,917 samples
- 557 features
- 424 targets
- final stored Sharpe-like score: `1.1911680408317646`
- mean correlation: `0.058018779359941784`
- standard deviation of correlation: `0.048707468107882276`
- recorded runtime: `15.128787553310394` minutes
- recorded device: `cuda`

The current repository does not identify a generator for this JSON strongly enough to reproduce the run from the artifact alone. Treat it as preserved experimental evidence, not as a competition submission result.

## Research files

The maintained experiment artifacts are under [`results/experiments/`](results/experiments/). Relevant implementation and research material is under:

- [`src/experiments/`](src/experiments/)
- [`research/`](research/)
- [`notebooks/`](notebooks/)
- [`docs/`](docs/)

The repository also contains competition input data under [`input/`](input/). Do not infer current competition rules, metric definitions, or leaderboard standing from repository notes; check the official Kaggle competition page.

## Reproducibility gap

The next research task is to connect every reported score to a runnable command and enough metadata to reproduce it. At minimum, a result should record:

- exact input files or hashes
- sample and target counts
- train/validation split
- random seed
- metric implementation
- package/runtime versions
- producing command or script
- output artifact path

Existing artifacts should be preserved while this provenance is reconstructed. Missing metadata should remain unknown rather than being inferred from older documentation.

## Documentation and Pages

GitHub Pages is currently served from the repository's `main:/docs` source:

https://kafka2306.github.io/mitsuikaggle/

The Pages site is independent of the research-artifact check workflow. The repository does not currently contain `mkdocs.yml`, so the previous `mkdocs gh-deploy` path was not a valid deployment route.

## Scope

This repository is for reproducible commodity-prediction research. A local improvement, a stored experiment artifact, and an official Kaggle leaderboard result are different claims and should remain separate until evidence connects them.
