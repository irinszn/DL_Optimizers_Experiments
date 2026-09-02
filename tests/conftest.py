import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import pytest

from src.config import (
    DataConfig,
    ExperimentConfig,
    GridSearchConfig,
    MLflowConfig,
    ModelConfig,
    NoiseTransformConfig,
    OptimizerConfig,
    RobustnessConfig,
    SchedulerConfig,
    TrainingConfig,
    TunerConfig,
)
from src.models.simple_cnn import SimpleCNN


NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE = 4


@pytest.fixture
def sample_tensor():
    """A single random 3-channel image tensor with fixed seed."""
    torch.manual_seed(0)
    return torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def sample_batch():
    """A batch of 16 random images and labels for 10-class classification."""
    torch.manual_seed(0)
    images = torch.rand(BATCH_SIZE * 4, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE * 4,))
    return images, labels


@pytest.fixture
def synthetic_dataloader(sample_batch):
    """DataLoader wrapping the sample_batch with batch_size=4."""
    images, labels = sample_batch
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture
def model():
    """SimpleCNN instance with fixed seed for reproducibility."""
    torch.manual_seed(0)
    return SimpleCNN(num_classes=NUM_CLASSES)


@pytest.fixture
def device():
    """CPU device for test runs."""
    return torch.device("cpu")


@pytest.fixture
def training_config():
    """Minimal training config for unit tests: 3 epochs, early stopping patience=2."""
    return TrainingConfig(
        epochs=3,
        batch_size=BATCH_SIZE,
        target_loss=0.5,
        criterion="CrossEntropyLoss",
        num_runs=1,
        save_model_mode="none",
        early_stopping_patience=2,
        early_stopping_metric="accuracy",
        scheduler=SchedulerConfig(name="constant", warmup_ratio=0.0, min_lr_ratio=0.1),
        use_tuner=False,
        tuner=TunerConfig(),
    )


@pytest.fixture
def sample_config_dict():
    """Full experiment config as a dict, suitable for YAML serialization."""
    return {
        "mlflow": {"experiment_name": "test_{model_name}_{dataset_name}"},
        "data": {
            "dataset_name": "TestDataset",
            "clean_data_path": "/tmp/test_data",
            "preprocessed_root_path": "/tmp/test_preprocessed",
            "scenario_folder_template": "Test_{scenario_name}",
            "num_classes": NUM_CLASSES,
            "num_workers": 0,
            "pin_memory": False,
            "debug_subset_size": 100,
        },
        "model": {"name": "SimpleCNN", "params": {"num_classes": NUM_CLASSES}},
        "training": {
            "epochs": 3,
            "batch_size": BATCH_SIZE,
            "target_loss": 0.5,
            "criterion": "CrossEntropyLoss",
            "num_runs": 1,
            "save_model_mode": "none",
            "early_stopping_patience": 2,
            "early_stopping_metric": "accuracy",
        },
        "robustness": {"trained_on_scenario": "no_noise"},
        "grid_search": {
            "optimizers": [
                {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}, "search_space": {}},
            ],
            "noise_scenarios": {
                "no_noise": [],
                "gaussian_0.05": [{"name": "GaussianNoiseAdder", "params": {"std": 0.05}}],
            },
        },
    }


@pytest.fixture
def sample_config_path(tmp_path, sample_config_dict):
    """Path to a temporary YAML config file created from sample_config_dict."""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config_dict, f)
    return str(config_file)


@pytest.fixture
def sample_experiment_config(sample_config_dict):
    """Fully parsed ExperimentConfig instance."""
    return ExperimentConfig(**sample_config_dict)
