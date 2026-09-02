import pytest
import torch
import torch.nn as nn

from src.models.simple_cnn import SimpleCNN
from src.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    def test_initial_state(self):
        """Fresh EarlyStopping should have default sentinel values."""
        es = EarlyStopping(patience=3, metric="accuracy")
        assert es.best_val_accuracy == -1.0
        assert es.best_epoch == 0
        assert es.best_model_state is None
        assert es.epochs_without_improvement == 0
        assert es.stopped_epoch == -1

    def test_improvement_saves_state(self):
        """First improvement should save model state and update best values."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3)
        metrics = {"accuracy": 50.0}
        should_stop = es.step(metrics, model, epoch=0)

        assert not should_stop
        assert es.best_val_accuracy == 50.0
        assert es.best_epoch == 1
        assert es.best_model_state is not None
        assert es.epochs_without_improvement == 0

    def test_no_improvement_increments_counter(self):
        """Worse accuracy should increment epochs_without_improvement."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3)
        es.step({"accuracy": 50.0}, model, epoch=0)
        es.step({"accuracy": 40.0}, model, epoch=1)

        assert es.epochs_without_improvement == 1
        assert es.best_val_accuracy == 50.0

    def test_stops_after_patience(self):
        """Should return True after patience epochs without improvement."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=2)
        es.step({"accuracy": 50.0}, model, epoch=0)
        es.step({"accuracy": 40.0}, model, epoch=1)
        should_stop = es.step({"accuracy": 30.0}, model, epoch=2)

        assert should_stop
        assert es.stopped_epoch == 3

    def test_patience_zero_never_stops(self):
        """With patience=0, training should never be stopped early."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=0)
        for i in range(20):
            should_stop = es.step({"accuracy": 50.0 - i}, model, epoch=i)
            assert not should_stop

    def test_improvement_resets_counter(self):
        """New best accuracy should reset the no-improvement counter."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3)
        es.step({"accuracy": 50.0}, model, epoch=0)
        es.step({"accuracy": 40.0}, model, epoch=1)
        assert es.epochs_without_improvement == 1

        es.step({"accuracy": 60.0}, model, epoch=2)
        assert es.epochs_without_improvement == 0
        assert es.best_val_accuracy == 60.0

    def test_deep_copy_model_state(self):
        """Saved model state should be a deep copy, unaffected by later weight changes."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3)
        es.step({"accuracy": 50.0}, model, epoch=0)

        original_state = {k: v.clone() for k, v in es.best_model_state.items()}

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        for key in original_state:
            assert torch.equal(es.best_model_state[key], original_state[key])

    def test_custom_metric(self):
        """EarlyStopping should track the configured metric, not always accuracy."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3, metric="f1_score")
        es.step({"f1_score": 80.0, "accuracy": 90.0}, model, epoch=0)
        assert es.best_val_accuracy == 80.0

    def test_missing_metric_defaults_zero(self):
        """Missing metric key in val_metrics should default to 0.0."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3, metric="precision")
        es.step({"accuracy": 80.0}, model, epoch=0)
        assert es.best_val_accuracy == 0.0

    def test_best_val_metrics_saved(self):
        """Full val_metrics dict should be stored on improvement."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3)
        metrics = {"accuracy": 60.0, "f1_score": 55.0}
        es.step(metrics, model, epoch=0)
        assert es.best_val_metrics == metrics

    def test_nan_metric_no_improvement(self):
        """NaN metric value should not count as improvement."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=3)
        es.step({"accuracy": float("nan")}, model, epoch=0)
        assert es.best_val_accuracy == -1.0
        assert es.best_model_state is None

    def test_monotonically_decreasing_stops_correctly(self):
        """Decreasing accuracy with patience=2 should stop at the right epoch."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=2)
        accuracies = [80.0, 70.0, 60.0]
        results = []
        for i, acc in enumerate(accuracies):
            results.append(es.step({"accuracy": acc}, model, epoch=i))

        assert results == [False, False, True]
        assert es.best_val_accuracy == 80.0
        assert es.best_epoch == 1
        assert es.stopped_epoch == 3

    def test_plateau_then_late_improvement(self):
        """Late improvement after plateau should reset counter and prevent stopping."""
        model = SimpleCNN(num_classes=10)
        es = EarlyStopping(patience=5)
        for i in range(3):
            should_stop = es.step({"accuracy": 50.0}, model, epoch=i)
            assert not should_stop
        should_stop = es.step({"accuracy": 60.0}, model, epoch=3)
        assert not should_stop
        assert es.epochs_without_improvement == 0
        assert es.best_val_accuracy == 60.0
        assert es.best_epoch == 4
