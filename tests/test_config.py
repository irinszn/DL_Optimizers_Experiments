import pytest
import yaml

from src.config import (
    ExperimentConfig,
    GridSearchConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TunerConfig,
    load_config,
)


class TestOptimizerConfig:
    def test_valid_optimizer(self):
        """Valid optimizer config should store name and params correctly."""
        opt = OptimizerConfig(name="SGD", params={"lr": 0.01, "momentum": 0.9})
        assert opt.name == "SGD"
        assert opt.params["lr"] == 0.01

    def test_missing_lr_raises(self):
        """Optimizer params without 'lr' should raise a validation error."""
        with pytest.raises(ValueError, match="must have 'lr' in params"):
            OptimizerConfig(name="Adam", params={"weight_decay": 0.001})

    def test_empty_params_raises(self):
        """Empty optimizer params should raise a validation error."""
        with pytest.raises(ValueError, match="must have 'lr' in params"):
            OptimizerConfig(name="SGD", params={})

    def test_search_space_defaults_empty(self):
        """search_space should default to an empty dict when not provided."""
        opt = OptimizerConfig(name="SGD", params={"lr": 0.1})
        assert opt.search_space == {}


class TestSchedulerConfig:
    def test_defaults(self):
        """Default scheduler should be 'constant' with no warmup."""
        cfg = SchedulerConfig()
        assert cfg.name == "constant"
        assert cfg.warmup_ratio == 0.0
        assert cfg.min_lr_ratio == 0.1


class TestTrainingConfig:
    def test_defaults(self):
        """TrainingConfig defaults should be sensible for single-run experiments."""
        cfg = TrainingConfig(epochs=10, batch_size=32, target_loss=0.5, criterion="CrossEntropyLoss")
        assert cfg.num_runs == 1
        assert cfg.save_model_mode == "best"
        assert cfg.early_stopping_patience == 0
        assert cfg.use_tuner is False

    def test_nested_scheduler(self):
        """Scheduler passed as a dict should be deserialized into SchedulerConfig."""
        cfg = TrainingConfig(
            epochs=10,
            batch_size=32,
            target_loss=0.5,
            criterion="CrossEntropyLoss",
            scheduler={"name": "cosine", "warmup_ratio": 0.1, "min_lr_ratio": 0.01},
        )
        assert cfg.scheduler.name == "cosine"
        assert cfg.scheduler.warmup_ratio == 0.1


class TestGridSearchConfig:
    def test_valid(self):
        """Grid search config should accept a list of optimizers and noise scenarios."""
        cfg = GridSearchConfig(
            optimizers=[OptimizerConfig(name="SGD", params={"lr": 0.01})],
            noise_scenarios={"no_noise": []},
        )
        assert len(cfg.optimizers) == 1
        assert "no_noise" in cfg.noise_scenarios


class TestExperimentConfig:
    def test_from_dict(self, sample_config_dict):
        """ExperimentConfig should be constructible from a nested dict."""
        cfg = ExperimentConfig(**sample_config_dict)
        assert cfg.model.name == "SimpleCNN"
        assert cfg.data.num_classes == 10
        assert cfg.training.epochs == 3
        assert len(cfg.grid_search.optimizers) == 1
        assert "no_noise" in cfg.grid_search.noise_scenarios

    def test_missing_required_field(self, sample_config_dict):
        """Missing required section should raise a validation error."""
        del sample_config_dict["training"]
        with pytest.raises(Exception):
            ExperimentConfig(**sample_config_dict)


class TestLoadConfig:
    def test_load_valid(self, sample_config_path):
        """Valid YAML config should load into a fully populated ExperimentConfig."""
        cfg = load_config(sample_config_path)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.model.name == "SimpleCNN"

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Malformed YAML should raise a parsing exception."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("not: a: valid: {yaml")
        with pytest.raises(Exception):
            load_config(str(bad_file))

    def test_load_missing_fields(self, tmp_path):
        """YAML with missing required sections should raise a validation exception."""
        incomplete = tmp_path / "incomplete.yaml"
        with open(incomplete, "w") as f:
            yaml.dump({"mlflow": {"experiment_name": "test"}}, f)
        with pytest.raises(Exception):
            load_config(str(incomplete))
