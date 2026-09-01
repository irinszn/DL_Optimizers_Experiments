# Comparison of Optimizers under Noisy Data Conditions

An experimental study comparing the **accuracy and robustness** of three optimization algorithms — **SGD**, **Adam**, and **LAMB** — under noisy data conditions. The goal is to determine which optimization strategy better maintains CNN accuracy when trained on data with different types and levels of distortion.

Experiments are conducted on the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset with artificially introduced **Gaussian** and **Salt & Pepper** noise at varying intensities. Models are evaluated using **Accuracy**, **Precision**, **Recall**, **F1-score**, and **convergence time**.

---

## Project Structure

```
DL_Optimizers_Experiments/
│
├── configs/
│   └── example_config.yaml       # Annotated config template
│
├── experiment_notebooks/
│   └── example.ipynb             # End-to-end run in Google Colab
│
├── src/
│   ├── config.py                 # Pydantic config schema and loader
│   ├── types.py                  # Type aliases (registries, factories)
│   ├── utils.py                  # Reproducibility utilities, scenario parsing
│   ├── data/
│   │   ├── noises.py             # GaussianNoiseAdder, SaltAndPepperNoiseAdder
│   │   └── processing.py         # Dataset generation and DataLoaders
│   ├── models/
│   │   └── simple_cnn.py         # SimpleCNN architecture
│   ├── experiment/
│   │   ├── runner.py             # ExperimentRunner — grid of optimizer × scenario runs
│   │   ├── robustness.py         # Cross-scenario robustness evaluation
│   │   ├── metrics.py            # Aggregation and summary tables
│   │   ├── mlflow_logger.py      # MLflow logging utilities
│   │   └── model_saver.py        # Model saving (best/all modes)
│   └── training/
│       ├── train.py              # Single epoch training loop
│       ├── evaluate.py           # Evaluation loop (Accuracy, Precision, Recall, F1)
│       ├── single_run.py         # Full training loop for one seed
│       ├── early_stopping.py     # Best-model tracking + patience-based stopping
│       ├── scheduler.py          # LR scheduler builder (constant, cosine, linear)
│       └── tuner.py              # Optuna hyperparameter tuner
│
├── tests/
├── run.py                        # Entry point
└── pyproject.toml
```

---

## Experimental Methodology

### Data Preparation

Noisy datasets are generated **once in advance** and saved to disk. For each noise scenario, every image from the clean dataset is transformed (resized to 128x128, noise applied) and stored as a separate folder. Generation is **idempotent** — existing folders are skipped. This ensures all optimizers and all runs train on exactly the same images.

The dataset is split into **train / val / test** (80/20, then train+val 85/15) with a fixed seed (`SPLIT_RANDOM_STATE = 42`) — all optimizers see identical data partitions.

The study evaluates optimizers in **two complementary scenarios**:

### Scenario 1 — Training on Noise (`--mode experiments`)

Each optimizer is trained **separately on each noise level** and tested on the **same noise level**. This measures how well an optimizer can learn from distorted data.

The full grid `optimizers × noise_scenarios` is evaluated. For example, SGD is trained on `gaussian_0.05` and tested on `gaussian_0.05`, then trained on `salt_pepper_0.03` and tested on `salt_pepper_0.03`, and so on. Each combination is run **3 times with different random seeds** to ensure statistical reliability.

### Scenario 2 — Robustness Evaluation (`--mode robustness`)

Each optimizer's best model, trained on **clean data** (`no_noise`), is evaluated on **all noise scenarios** without retraining. This measures how robust the learned representations are to unseen noise.

The result is a cross-scenario accuracy matrix: one row per noise level, one column per optimizer.

### Statistical Reliability

- Each `(optimizer, scenario)` combination is trained `num_runs` times (default: 3) with different random seeds
- The same set of seeds is shared across all combinations — every optimizer trains on the same seed sequence for fair comparison
- Metrics are aggregated: **mean**, **std**, and **95% confidence interval** (Student's t-distribution, appropriate for small `num_runs`)
- Convergence time (wall-clock seconds to reach `target_loss` on **training loss**) is tracked separately for runs that converge

### Hyperparameter Selection

Each optimizer has its own `lr`, `weight_decay`, and other parameters specified in the config. Two modes are available:

**Manual** (`use_tuner: false`): parameters from the `params` section of each optimizer are used directly. Note: parameters not explicitly set in `params` (e.g. `betas`, `weight_decay` for Adam) will use library defaults.

**Automatic** (`use_tuner: true`): Optuna searches for optimal parameters per optimizer:

- **Per-scenario tuning** — Optuna runs independently on selected `tune_scenarios` (e.g. `no_noise`, `gaussian_0.05`, `salt_pepper_0.03`)
- **Nearest neighbor** — scenarios not in `tune_scenarios` receive parameters from the nearest tuned scenario of the same noise type (matched by noise level)
- **Fixed params** — parameters in `params` but not in `search_space` (e.g. `nesterov: true`) are not tuned, they pass through as constants
- **Reproducibility** — every Optuna trial starts with `set_random_seed(42)`, so the winning trial is determined by hyperparameters, not by a lucky initialization
- All found hyperparameters are logged to console, `experiment.log`, and MLflow

### LR Scheduler

A **single scheduler** config applies uniformly to all optimizers (fair comparison):

| Name       | Behavior                                              |
|------------|-------------------------------------------------------|
| `constant` | LR stays the same throughout training (default)       |
| `cosine`   | CosineAnnealingLR, decays to `lr × min_lr_ratio`     |
| `linear`   | LinearLR, decays to `lr × min_lr_ratio`               |

Scheduler operates at **epoch level**. Optional **warmup** is available via `warmup_ratio`: the first `max(1, total_epochs × warmup_ratio)` epochs use LinearLR warmup from `lr × 1e-3` to `lr`, then the main schedule takes over (`SequentialLR`). Default `warmup_ratio: 0.0` disables warmup.

### Early Stopping

Monitors a configurable validation metric (`accuracy`, `f1_score`, `precision`, `recall`). If the metric does not improve for `early_stopping_patience` epochs, training stops early. The best model checkpoint is always tracked regardless of patience setting (`patience: 0` disables stopping but still records the best epoch).

---

## Supported Optimizers

| Optimizer | Source |
|-----------|--------|
| SGD       | `torch.optim` |
| Adam      | `torch.optim` |
| LAMB      | `torch-optimizer` |

To add an optimizer, register it in `OPTIMIZER_REGISTRY` in `run.py`, add a `_suggest_*_params` method to `HyperparameterTuner`, and configure it in the YAML config.

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
  debug_subset_size: 100  # use a small subset for quick test runs (remove for full run)

model:
  name: "SimpleCNN"
  params:
    num_classes: 10

training:
  epochs: 12
  batch_size: 64
  target_loss: 0.4
  criterion: "CrossEntropyLoss"
  num_runs: 3             # runs per experiment for averaging
  save_model_mode: "best" # best | all | none
  early_stopping_patience: 3
  early_stopping_metric: "accuracy"
  scheduler:
    name: "constant"      # constant | cosine | linear
    warmup_ratio: 0.0
    min_lr_ratio: 0.1
  use_tuner: false
  tuner:
    n_trials: 50
    tune_scenarios:
      - "no_noise"

robustness:
  trained_on_scenario: "no_noise"

grid_search:
  optimizers:
    - name: "SGD"
      params:
        lr: 0.01
        momentum: 0.9
        nesterov: true
        weight_decay: 0.0001
      search_space:       # ranges for Optuna (used when use_tuner: true)
        lr_log10: [-4, 0]
        momentum: [0.5, 0.99]
        wd_log10: [-6, -3.3]
    - name: "Adam"
      params:
        lr: 0.001
      search_space:
        lr_log10: [-5, -2]
        beta1: [0.8, 0.99]
        beta2: [0.9, 0.999]
        wd_log10: [-6, -3]

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

Config is validated at startup via Pydantic — missing or mistyped fields (including missing `lr` in optimizer params) produce a clear error immediately.

---

## Running

### Locally / on a server

```bash
# Full pipeline: generate noisy datasets + experiments + robustness
python run.py --config configs/your_config.yaml --mode all

# Individual stages
python run.py --mode generate      # Generate noisy datasets (idempotent, skips existing)
python run.py --mode experiments   # Scenario 1: train on each noise level, test on same noise
python run.py --mode robustness    # Scenario 2: evaluate clean-trained models on all noise levels
```

### Google Colab

Open `experiment_notebooks/example.ipynb`. The notebook covers:

1. Clone the repo and install dependencies
2. Connect Google Drive and MLflow
3. Download Animals-10 from Kaggle
4. Generate noisy datasets (writes to local SSD first, then copies to Drive)
5. (Optional) Run Optuna hyperparameter tuning
6. Run experiments and robustness evaluation

---

## Experiment Tracking

All runs are logged to **MLflow** with a nested structure:

```
SGD_no_noise (parent)
├── SGD_no_noise_run_1 (child, seed 1)
├── SGD_no_noise_run_2 (child, seed 2)
└── SGD_no_noise_run_3 (child, seed 3)
```

**Parent run**: aggregated test metrics (mean, std, CI 95%), convergence stats, tuned hyperparameters, best model artifact.

**Child runs**: per-epoch `val_accuracy`, `val_f1_score`, `epoch_loss`, `convergence_time`, `best_epoch`, test metrics.

---

## Reproducibility

- Train/val/test split is fixed via `SPLIT_RANDOM_STATE = 42` — identical across all optimizer runs
- Each of the `num_runs` runs uses a different randomly generated seed, logged to MLflow
- The same seed sequence is reused for every `(optimizer, scenario)` combination
- `torch.backends.cudnn.deterministic = True` is set when CUDA is available
- Optuna trials use a fixed `TUNER_SEED = 42` for model initialization
