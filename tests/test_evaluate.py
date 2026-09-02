import torch
import pytest

from src.models.simple_cnn import SimpleCNN
from src.training.evaluate import evaluate_model


class TestEvaluateModel:
    def test_returns_all_metrics(self, model, synthetic_dataloader, device):
        """Result dict should contain accuracy, precision, recall, f1_score."""
        metrics = evaluate_model(model, synthetic_dataloader, device)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_metrics_range(self, model, synthetic_dataloader, device):
        """All metric values should be in [0, 100] range."""
        metrics = evaluate_model(model, synthetic_dataloader, device)
        for name, value in metrics.items():
            assert 0.0 <= value <= 100.0, f"{name} out of range: {value}"

    def test_perfect_predictions(self, device):
        """Model with perfect predictions should achieve 100% accuracy."""

        class PerfectModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 10)
                for i in range(batch_size):
                    out[i, i % 10] = 100.0
                return out

        model = PerfectModel()
        images = torch.rand(10, 3, 32, 32)
        labels = torch.arange(10)
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        metrics = evaluate_model(model, loader, device)
        assert metrics["accuracy"] == pytest.approx(100.0)

    def test_model_set_to_eval(self, model, synthetic_dataloader, device):
        """evaluate_model should switch the model to eval mode."""
        model.train()
        evaluate_model(model, synthetic_dataloader, device)
        assert not model.training

    def test_known_misclassifications(self, device):
        """Model always predicting class 0 on balanced 5-class data should get 20% accuracy."""

        class AlwaysZeroModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 5)
                out[:, 0] = 100.0
                return out

        images = torch.rand(10, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        metrics = evaluate_model(AlwaysZeroModel(), loader, device)
        assert metrics["accuracy"] == pytest.approx(20.0)
        assert metrics["precision"] == pytest.approx(4.0, abs=0.1)

    def test_multi_batch_accumulation(self, device):
        """Metrics should accumulate correctly across multiple batches (batch_size=3, 10 samples)."""

        class AlwaysClass0Model(torch.nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 3)
                out[:, 0] = 100.0
                return out

        images = torch.rand(10, 3, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=3)

        metrics = evaluate_model(AlwaysClass0Model(), loader, device)
        assert metrics["accuracy"] == pytest.approx(100.0)

    def test_metrics_are_percentages(self, device):
        """All metrics should be on 0-100 scale, not 0-1."""

        class PerfectModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.size(0)
                out = torch.zeros(batch_size, 10)
                for i in range(batch_size):
                    out[i, i % 10] = 100.0
                return out

        images = torch.rand(10, 3, 32, 32)
        labels = torch.arange(10)
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        metrics = evaluate_model(PerfectModel(), loader, device)
        for name in ("accuracy", "precision", "recall", "f1_score"):
            assert metrics[name] == pytest.approx(100.0), f"{name} should be 100.0, got {metrics[name]}"
