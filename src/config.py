from typing import Any

import yaml
from pydantic import BaseModel, Field


class MLflowConfig(BaseModel):
    experiment_name: str


class DataConfig(BaseModel):
    dataset_name: str
    clean_data_path: str
    preprocessed_root_path: str
    scenario_folder_template: str
    num_classes: int
    num_workers: int = 2
    pin_memory: bool = False
    debug_subset_size: int | None = None


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(BaseModel):
    name: str = "constant"  # constant | cosine | linear
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.1


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    target_loss: float
    criterion: str
    num_runs: int = 1
    save_model_mode: str = "best"
    early_stopping_patience: int = 0
    early_stopping_metric: str = "accuracy"
    scheduler: SchedulerConfig = SchedulerConfig()


class RobustnessConfig(BaseModel):
    trained_on_scenario: str


class NoiseTransformConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class GridSearchConfig(BaseModel):
    optimizers: list[OptimizerConfig]
    noise_scenarios: dict[str, list[NoiseTransformConfig]]


class ExperimentConfig(BaseModel):
    mlflow: MLflowConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    robustness: RobustnessConfig
    grid_search: GridSearchConfig


def load_config(config_path: str) -> ExperimentConfig:
    """Loads and validates a YAML config file, returning a typed ExperimentConfig."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)
