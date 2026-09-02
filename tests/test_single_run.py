import torch
import pytest

from src.config import SchedulerConfig, TrainingConfig
from src.training.single_run import SingleRunResult, train_single_run


class TestSingleRunResult:
    def test_dataclass_fields(self):
        """All SingleRunResult fields should be accessible after construction."""
        result = SingleRunResult(
            epoch_losses=[1.0, 0.5],
            val_metrics_history=[{"accuracy": 50.0}, {"accuracy": 60.0}],
            best_val_metrics={"accuracy": 60.0},
            best_model_state=None,
            best_val_accuracy=60.0,
            best_epoch=2,
            convergence_time=1.5,
            stopped_epoch=-1,
        )
        assert result.best_epoch == 2
        assert result.convergence_time == 1.5
        assert len(result.epoch_losses) == 2


class TestTrainSingleRun:
    def test_basic_run(self, model, synthetic_dataloader, training_config, device):
        """Basic training run should return a valid SingleRunResult with correct shapes."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        result = train_single_run(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=synthetic_dataloader,
            val_loader=synthetic_dataloader,
            training_config=training_config,
            device=device,
        )

        assert isinstance(result, SingleRunResult)
        assert len(result.epoch_losses) <= training_config.epochs
        assert len(result.val_metrics_history) == len(result.epoch_losses)
        assert result.best_val_accuracy >= 0
        assert result.best_epoch >= 1
        assert result.best_model_state is not None

    def test_convergence_time_tracked(self, model, synthetic_dataloader, device):
        """With a high target_loss, convergence should be detected and time recorded."""
        config = TrainingConfig(
            epochs=5,
            batch_size=4,
            target_loss=999.0,
            criterion="CrossEntropyLoss",
            early_stopping_patience=0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        result = train_single_run(model, optimizer, criterion, synthetic_dataloader, synthetic_dataloader, config, device)
        assert result.convergence_time is not None
        assert result.convergence_time >= 0

    def test_no_convergence(self, model, synthetic_dataloader, device):
        """With an impossibly low target_loss, convergence_time should be None."""
        config = TrainingConfig(
            epochs=2,
            batch_size=4,
            target_loss=0.0001,
            criterion="CrossEntropyLoss",
            early_stopping_patience=0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        result = train_single_run(model, optimizer, criterion, synthetic_dataloader, synthetic_dataloader, config, device)
        assert result.convergence_time is None

    def test_early_stopping_triggers(self, model, synthetic_dataloader, device):
        """Early stopping with patience=2 and tiny lr should stop well before 100 epochs."""
        config = TrainingConfig(
            epochs=100,
            batch_size=4,
            target_loss=0.001,
            criterion="CrossEntropyLoss",
            early_stopping_patience=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        result = train_single_run(model, optimizer, criterion, synthetic_dataloader, synthetic_dataloader, config, device)
        assert len(result.epoch_losses) <= 10

    def test_with_scheduler(self, model, synthetic_dataloader, device):
        """Training with a cosine scheduler should complete without errors."""
        config = TrainingConfig(
            epochs=3,
            batch_size=4,
            target_loss=0.5,
            criterion="CrossEntropyLoss",
            early_stopping_patience=0,
            scheduler=SchedulerConfig(name="cosine", warmup_ratio=0.0, min_lr_ratio=0.01),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        result = train_single_run(model, optimizer, criterion, synthetic_dataloader, synthetic_dataloader, config, device)
        assert isinstance(result, SingleRunResult)
        assert len(result.epoch_losses) == 3

    def test_best_epoch_within_range(self, model, synthetic_dataloader, training_config, device):
        """Best epoch should be between 1 and the total number of epochs."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        result = train_single_run(model, optimizer, criterion, synthetic_dataloader, synthetic_dataloader, training_config, device)
        assert 1 <= result.best_epoch <= training_config.epochs

    def test_val_metrics_history_has_all_keys(self, model, synthetic_dataloader, training_config, device):
        """Each epoch's validation metrics should contain accuracy, precision, recall, f1_score."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        result = train_single_run(model, optimizer, criterion, synthetic_dataloader, synthetic_dataloader, training_config, device)
        expected_keys = {"accuracy", "precision", "recall", "f1_score"}
        for epoch_metrics in result.val_metrics_history:
            assert expected_keys.issubset(epoch_metrics.keys())
