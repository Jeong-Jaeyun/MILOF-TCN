# MILOF-TCN

MiLOF-TCN is an edge-fog anomaly detection pipeline for hourly electric vehicle charging demand. The project combines a lightweight streaming MiLOF-style edge detector with a fog-side TCN autoencoder, and includes statistical and Isolation Forest baselines for comparison.

The runnable code is in [`ev-anomaly`](./ev-anomaly).

## Overview

This repository implements a two-stage anomaly detection workflow.

1. Raw hourly EV charging records are converted into a long time-series table.
2. Synthetic anomalies are injected for controlled evaluation.
3. Baseline detectors are evaluated on the same train, validation, and test splits.
4. The proposed MiLOF-TCN pipeline uses an edge detector for early routing and a TCN autoencoder for reconstruction-based anomaly scoring.
5. Metrics, curves, diagnostic tables, plots, and trained model artifacts are saved under `results/`.

The main evaluation metrics include precision, recall, F1, PR-AUC, ROC-AUC, MCC, balanced accuracy, false alarms per day, and detection delay.

## Repository Structure

```text
MILOF-TCN-main/
|-- README.md
`-- ev-anomaly/
    |-- configs/                 # Experiment configuration files
    |-- data/
    |   |-- raw/                 # Raw CSV input
    |   `-- processed/           # Generated parquet files
    |-- scripts/                 # Pipeline, experiment, and artifact scripts
    |-- src/
    |   |-- common/              # IO, logging, metrics, plotting, seeding
    |   |-- dataset/             # Preprocessing, splitting, anomaly injection, windows
    |   |-- evaluation/          # Metric and latency evaluation
    |   |-- models/              # MiLOF edge detector, TCN-AE, baselines
    |   `-- training/            # TCN training and threshold utilities
    |-- results/                 # Generated experiment outputs
    |-- revision_runs/           # Isolated multi-seed revision runs
    `-- result_for_revision/     # Revision figures and tables
```

## Requirements

Recommended environment:

- Python 3.10 or newer
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- PyYAML
- pyarrow

Install the dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib PyYAML pyarrow
```

If you use a CUDA-enabled GPU, install the PyTorch build that matches your CUDA environment before running the training scripts.

## Data

The default configuration expects the raw CSV at:

```text
ev-anomaly/data/raw/ElecCar_20240930.csv
```

The preprocessing code expects a UTF-8 CSV containing:

- a date column
- a charging-mode column
- 24 hourly columns

Column names and charging-mode labels are configured in [`ev-anomaly/configs/base.yaml`](./ev-anomaly/configs/base.yaml). If you use another dataset, update the `paths`, `data`, and `split` sections in the config file before running the pipeline.

## Quick Start

Run commands from the `ev-anomaly` directory:

```bash
cd MILOF-TCN-main/ev-anomaly
```

Prepare the time-series dataset:

```bash
python scripts/prepare_data.py --config configs/base.yaml
```

Create train, validation, and test splits with injected anomalies:

```bash
python scripts/inject_anomalies.py --config configs/base.yaml
```

Run the baseline models:

```bash
python scripts/run_baselines.py --config configs/base.yaml
```

Run the proposed MiLOF-TCN model:

```bash
python scripts/run_milof_tcn.py --config configs/base.yaml
```

## Outputs

Preprocessed files are saved under:

```text
ev-anomaly/data/processed/
```

Injected split files are saved under:

```text
ev-anomaly/data/processed/injected/
|-- train.parquet
|-- train_labels.parquet
|-- val.parquet
|-- val_labels.parquet
|-- test.parquet
`-- test_labels.parquet
```

Each experiment creates a timestamped run directory under `ev-anomaly/results/`, for example:

```text
results/
|-- baseline_stats_<timestamp>/
|-- baseline_iforest_<timestamp>/
`-- milof_tcn_<timestamp>/
```

Typical run artifacts include:

- `config_resolved.yaml`: configuration used for the run
- `metrics.json`: final evaluation metrics
- `curves.csv`: window-level scores and predictions
- `figures/score.png`: anomaly score plot
- `figures/pred.png`: prediction plot
- `artifacts/tcn_ae.pt`: trained TCN autoencoder checkpoint
- `window_routing_diagnostics.csv`: edge-fog routing diagnostics
- `edge_stats.json`: edge detector and routing statistics

## Configuration

Main configuration file:

```text
ev-anomaly/configs/base.yaml
```

Important sections:

- `paths`: raw data, processed data, and result directories
- `data`: input column names and charging-mode labels
- `features`: selected features, calendar features, and rolling features
- `split`: train, validation, and test date ranges
- `injection`: synthetic anomaly rate and anomaly type settings
- `windows`: window length, stride, and window label rule
- `tcn`: TCN autoencoder architecture and training settings
- `threshold`: validation-based anomaly thresholding rule
- `edge`: streaming MiLOF-style edge detector settings
- `edge_window`: default edge-window forwarding threshold
- `routing`: optional edge-fog routing behavior

## Additional Experiments

Run a small hyperparameter sweep:

```bash
python scripts/run_sweep.py --config configs/base.yaml
```

The sweep summary is saved to:

```text
results/sweep_summary.csv
```

Run a multi-seed revision pipeline in an isolated workspace:

```bash
python scripts/run_revision_pipeline.py --config configs/base.yaml --seeds 3,13,23,33,43 --suite_tag rev1
```

This creates:

```text
revision_runs/rev1/
|-- configs/
|-- logs/
|-- processed/
|-- results/
`-- pipeline_manifest.json
```

Build revision tables and figures from completed runs:

```bash
python scripts/build_revision_artifacts.py --results_root results --output_dir results/revision_artifacts
```

## Model Components

- `ContextZScoreBaseline`: context-aware statistical z-score baseline.
- `IsolationForestBaseline`: point-level Isolation Forest baseline.
- `MiLOFEdgeDetector`: streaming micro-cluster detector that computes LOF-like edge anomaly scores.
- `TCNAutoEncoder`: temporal convolutional autoencoder that scores windows by reconstruction error.

## Notes for GitHub Upload

Generated experiment folders can become large. Before publishing, decide whether to track or ignore:

- `ev-anomaly/data/raw/`
- `ev-anomaly/data/processed/`
- `ev-anomaly/results/`
- `ev-anomaly/revision_runs/`
- `ev-anomaly/result_for_revision/`

If the raw data is private or licensed, do not upload it. Keep the folder structure and provide instructions for placing the CSV at the expected path.
