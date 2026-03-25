# ASTD Project 3 - ROCKET

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run experiments

```bash
# Run all datasets
python src/benchmark.py

# Run a single dataset (useful for distributing across machines)
python src/benchmark.py --dataset Computers
```

Runs 5-fold stratified CV for 11 classifiers on 8 UCR datasets. Results are saved to `results/benchmark_results.csv`.

### 2. Analyze results

```bash
python src/analysis.py
```

Generates all outputs in `results/`:
- `summary_table.csv` — mean accuracy per dataset/classifier
- `accuracy_heatmap.pdf` — color-coded accuracy matrix
- `accuracy_boxplots.pdf` — per-dataset accuracy distributions
- `timing_barplot.pdf` — training time comparison (log scale)
- `cd_diagram.pdf` — critical difference diagram (Nemenyi post-hoc)
- `statistical_tests.txt` — Friedman test and pairwise comparisons

## Classifiers

| Name | Algorithm |
|---|---|
| ROCKET | Random Convolutional Kernel Transform |
| 1NN-ED | 1-Nearest Neighbor (Euclidean) |
| 1NN-DTW | 1-Nearest Neighbor (Dynamic Time Warping) |
| ST | Shapelet Transform |
| BOSS | BOSS Ensemble |
| TSF | Time Series Forest |
| InceptionTime | InceptionTime Deep Learning |
| catch22 | Canonical Time-series Characteristics |

## Datasets

Computers, ECG200, DistalPhalanxTW, Worms, Earthquakes

## Project Structure

```
src/
  config.py       # Datasets, classifiers, CV settings
  benchmark.py    # Experiment runner
  analysis.py     # Visualization and statistical analysis
docs/             # Project specification and references
results/          # Generated outputs (after running)
```
