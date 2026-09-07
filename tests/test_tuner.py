import pytest
import torch
from optuna.trial import FixedTrial

from src.config import SchedulerConfig
from src.models.simple_cnn import SimpleCNN
from src.training.tuner import HyperparameterTuner


@pytest.fixture
def tuner(synthetic_dataloader):
    """SGD tuner with minimal config for unit tests."""
    return HyperparameterTuner(
        model_class=SimpleCNN,
        model_params={"num_classes": 10},
        optimizer_name="SGD",
        optimizer_factory=torch.optim.SGD,
        search_space={},
        train_loader=synthetic_dataloader,
        val_loader=synthetic_dataloader,
        epochs_per_trial=2,
        early_stopping_patience=1,
    )


class TestHyperparameterTuner:
    def test_unsupported_optimizer_raises(self, synthetic_dataloader):
        """Unsupported optimizer name should raise ValueError at construction."""
        with pytest.raises(ValueError, match="doesn't support"):
            HyperparameterTuner(
                model_class=SimpleCNN,
                model_params={"num_classes": 10},
                optimizer_name="Unsupported",
                optimizer_factory=torch.optim.SGD,
                search_space={},
                train_loader=synthetic_dataloader,
                val_loader=synthetic_dataloader,
            )

    def test_range_with_custom_space(self, tuner):
        """Custom search_space should override default range bounds."""
        tuner.search_space = {"lr_log10": [-3, -1]}
        lo, hi = tuner._range("lr_log10", -4, 0)
        assert lo == -3
        assert hi == -1

    def test_range_with_defaults(self, tuner):
        """Missing key in search_space should fall back to default bounds."""
        lo, hi = tuner._range("nonexistent", -5, -2)
        assert lo == -5
        assert hi == -2

    def test_suggest_sgd_params(self, tuner):
        """SGD param suggestion should produce lr, momentum, and weight_decay from trial values."""
        trial = FixedTrial({"lr_log10": -2.0, "momentum": 0.9, "wd_log10": -4.0})
        params = tuner._suggest_sgd_params(trial)

        assert params["lr"] == pytest.approx(0.01)
        assert params["momentum"] == 0.9
        assert params["weight_decay"] == pytest.approx(1e-4)
        assert 0 <= params["momentum"] < 1

    def test_suggest_adam_params(self, synthetic_dataloader):
        """Adam param suggestion should produce lr, betas tuple, and weight_decay."""
        tuner = HyperparameterTuner(
            model_class=SimpleCNN,
            model_params={"num_classes": 10},
            optimizer_name="Adam",
            optimizer_factory=torch.optim.Adam,
            search_space={},
            train_loader=synthetic_dataloader,
            val_loader=synthetic_dataloader,
        )
        trial = FixedTrial({"lr_log10": -3.0, "beta1": 0.9, "beta2": 0.999, "wd_log10": -5.0})
        params = tuner._suggest_adam_params(trial)

        assert params["lr"] == pytest.approx(1e-3)
        assert isinstance(params["betas"], tuple)
        assert len(params["betas"]) == 2
        assert 0 < params["betas"][0] < 1
        assert 0 < params["betas"][1] < 1
        assert params["weight_decay"] > 0

    def test_suggest_lamb_params(self, synthetic_dataloader):
        """LAMB param suggestion should produce lr and betas within valid ranges."""
        import torch_optimizer

        tuner = HyperparameterTuner(
            model_class=SimpleCNN,
            model_params={"num_classes": 10},
            optimizer_name="LAMB",
            optimizer_factory=torch_optimizer.Lamb,
            search_space={},
            train_loader=synthetic_dataloader,
            val_loader=synthetic_dataloader,
        )
        trial = FixedTrial({"lr_log10": -2.5, "beta1": 0.9, "beta2": 0.95, "wd_log10": -3.0})
        params = tuner._suggest_lamb_params(trial)

        assert params["lr"] == pytest.approx(10**-2.5)
        assert 0 < params["betas"][0] < 1
        assert 0 < params["betas"][1] < 1

    def test_fixed_params_merged(self, synthetic_dataloader):
        """Fixed params should be merged into suggested params."""
        tuner = HyperparameterTuner(
            model_class=SimpleCNN,
            model_params={"num_classes": 10},
            optimizer_name="SGD",
            optimizer_factory=torch.optim.SGD,
            search_space={},
            train_loader=synthetic_dataloader,
            val_loader=synthetic_dataloader,
            fixed_params={"nesterov": True},
        )
        trial = FixedTrial({"lr_log10": -2.0, "momentum": 0.9, "wd_log10": -4.0})
        params = tuner._suggest_sgd_params(trial)
        assert params["nesterov"] is True

    def test_tune_runs(self, tuner):
        """Running tune with 2 trials should return a dict with valid lr and momentum."""
        best_params = tuner.tune(n_trials=2, timeout=60)
        assert "lr" in best_params
        assert "momentum" in best_params
        assert best_params["lr"] > 0
