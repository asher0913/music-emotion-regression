# Music Emotion Regression

A reproducible machine-learning study of song-level valence and arousal on the
DEAM dataset. The notebook turns 520 aggregated openSMILE acoustic descriptors
into a leakage-safe regression benchmark with classical, ensemble, and
histogram-based boosting models.

## Results

The best held-out results came from `HistGradientBoostingRegressor` on a fixed
20% test split (361 songs):

| Target | MAE | RMSE | R² | 95% bootstrap RMSE interval |
| --- | ---: | ---: | ---: | ---: |
| Arousal | 0.593 | 0.765 | 0.604 | 0.706–0.823 |
| Valence | 0.676 | 0.872 | 0.433 | 0.800–0.937 |

The analysis compares eight model families and 74 hyperparameter
configurations with five-fold cross-validation. It also includes:

- target-distribution, correlation, and PCA exploration;
- train-only variance filtering, imputation, and standardization;
- Random Forest and Gradient Boosting feature importance;
- bootstrap confidence intervals and paired Wilcoxon comparisons;
- permutation importance and residual diagnostics;
- a per-rater noise-ceiling estimate and Ridge bias–variance curve;
- learning curves for diagnosing data- versus feature-limited performance.

## Reproduce the notebook

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/music_emotion_regression.ipynb
```

Run the first notebook cell to download the 18.7 MB feature table from the
companion [dataset release](https://github.com/asher0913/deam-song-level-features/releases/tag/1).
The repository versions only a privacy-preserving per-song summary of rater
counts and annotation variability under `data/`; individual rater IDs are not
retained.

Notebook outputs are retained so the full analysis and reported metrics can be
reviewed on GitHub without rerunning the multi-model search.

## Repository layout

```text
notebooks/music_emotion_regression.ipynb  # executable analysis and outputs
data/deam_rater_summary.csv              # per-song agreement statistics
outputs/                                  # generated when the notebook runs
```

## Data attribution

The analysis uses the Database for Emotional Analysis in Music (DEAM): Aljanaki,
Yang, and Soleymani, “Developing a benchmark for emotional analysis of music,”
*PLOS ONE* 12(3), 2017. The repository contains derived analysis code and one
compact annotation table; audio is not redistributed.
