# Comparison of Optimizers under Noisy Data Conditions

An experimental study comparing the **accuracy and robustness** of three optimization algorithms — **SGD**, **Adam**, and **LAMB** — under noisy data conditions. The goal is to determine which optimization strategy better maintains CNN accuracy when trained on data with different types and levels of distortion.

Experiments are conducted on the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset with artificially introduced **Gaussian** and **Salt & Pepper** noise at varying intensities. Models are evaluated using **Accuracy**, **Precision**, **Recall**, **F1-score**, and **convergence time**.

---

## Project Structure

```
DL_Optimizers_Experiments/
│
├── configs/
│   └── example_config.yaml       # Template config — copy and fill in your paths
│
├── experiment_notebooks/
│   └── example.ipynb             # End-to-end run in Google Colab
│
├── src/
│   ├── config.py                 # Pydantic config schema and loader
│   ├── utils.py                  # Reproducibility utilities
│   ├── data/
│   │   ├── noises.py             # GaussianNoiseAdder, SaltAndPepperNoiseAdder
│   │   └── processing.py         # Dataset generation and DataLoaders
│   ├── models/
│   │   └── simple_cnn.py         # SimpleCNN architecture
│   ├── experiment/
│   │   ├── runner.py             # ExperimentRunner — grid of optimizer × scenario runs
│   │   ├── robustness.py         # Cross-scenario robustness evaluation
│   │   └── metrics.py            # Aggregation and summary tables
│   └── training/
│       ├── train.py              # Single epoch training loop
│       ├── evaluate.py           # Evaluation loop
│       └── tuner.py              # Optuna hyperparameter tuner
│
├── tests/
├── run.py                        # Entry point
└── pyproject.toml
```

---

## Supported Optimizers

| Optimizer | Status |
|-----------|--------|
| SGD       | supported |
| Adam      | supported |
| LAMB      | supported |

To add an optimizer, register it in `OPTIMIZER_REGISTRY` in `run.py`:

```python
OPTIMIZER_REGISTRY = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}
```

---

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/irinszn/DL_Optimizers_Experiments.git
cd DL_Optimizers_Experiments
uv sync
```

---

## Configuration

Copy `configs/example_config.yaml` and fill in your paths:

```yaml
mlflow:
  experiment_name: "{model_name}_{dataset_name}"

data:
  dataset_name: "Animals10"
  clean_data_path: "path/to/raw/data"
  preprocessed_root_path: "path/to/preprocessed/data"
  scenario_folder_template: "Animals10_{scenario_name}"
  num_classes: 10
  num_workers: 2
  pin_memory: false       # set to true on GPU

model:
  name: "SimpleCNN"
  params:
    num_classes: 10

training:
  epochs: 12
  batch_size: 64
  learning_rate: 0.001
  target_loss: 0.4
  criterion: "CrossEntropyLoss"
  num_runs: 3             # runs per experiment for averaging
  save_model_mode: "best" # best | all | none
  early_stopping_patience: 3

robustness:
  trained_on_scenario: "no_noise"  # scenario used to train the reference model

grid_search:
  optimizers:
    - name: "SGD"
      params:
        momentum: 0.9
    - name: "Adam"
      params: {}

  noise_scenarios:
    no_noise: []
    gaussian_0.05:
      - name: "GaussianNoiseAdder"
        params:
          mean: 0.0
          std: 0.05
    salt_pepper_0.03:
      - name: "SaltAndPepperNoiseAdder"
        params:
          amount: 0.03
```

Config is validated at startup via Pydantic — missing or mistyped fields produce a clear error immediately.

---

## Running

### Locally / on a server

```bash
# Run full pipeline: experiments + robustness evaluation
python run.py --config configs/your_config.yaml

# Run only training
python run.py --config configs/your_config.yaml --mode experiments

# Run only robustness evaluation (requires trained models in MLflow)
python run.py --config configs/your_config.yaml --mode robustness
```

### Google Colab

Open `experiment_notebooks/example.ipynb`. The notebook covers:

1. Clone the repo and install dependencies
2. Connect Google Drive and MLflow
3. Download Animals-10 from Kaggle
4. Generate noisy datasets and upload to Drive
5. (Optional) Run Optuna hyperparameter tuning
6. Run experiments and robustness evaluation

---

## Experiment Pipeline

**Experiments** (`run_experiments`):

- Iterates over all `noise_scenarios × optimizers` combinations
- Each combination is run `num_runs` times with different seeds for statistical reliability
- Per run: trains the model, applies early stopping, restores best-epoch checkpoint for test evaluation
- Logs per-epoch metrics, convergence time, and aggregated statistics to MLflow

**Robustness evaluation** (`run_robustness`):

- Loads the best model trained on `robustness.trained_on_scenario` from MLflow
- Evaluates it on every noise scenario in `grid_search.noise_scenarios`
- Produces a cross-scenario accuracy matrix per optimizer

---

## Hyperparameter Tuning

Optuna-based tuner is available via `HyperparameterTuner` in `src/training/tuner.py`. Supports SGD, Adam, and LAMB search spaces. Typically run before the main experiment to find optimal hyperparameters per optimizer.

---

## Experiment Tracking

All runs are logged to **MLflow**:

- Per-epoch `val_accuracy`, `val_f1_score`, `epoch_loss`
- `convergence_time`, `best_epoch`, `best_val_accuracy`
- Aggregated test metrics with mean, std, and 95% confidence intervals
- Best model artifact saved according to `save_model_mode`

---

## Reproducibility

- Train/val/test split is fixed via `SPLIT_RANDOM_STATE` — identical across all optimizer runs
- Each of the `num_runs` runs uses a different randomly generated seed, logged to MLflow
- `torch.backends.cudnn.deterministic = True` is set when CUDA is available
