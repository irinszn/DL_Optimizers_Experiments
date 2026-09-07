from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from src.experiment.model_saver import save_best_overall_model, save_run_model
from src.models.simple_cnn import SimpleCNN


@pytest.fixture
def mock_signature():
    """Mock MLflow model signature for saver tests."""
    return MagicMock()


class TestSaveRunModel:
    @patch("src.experiment.model_saver.mlflow")
    def test_logs_model(self, mock_mlflow, mock_signature):
        """save_run_model should log the model to MLflow with the correct artifact name."""
        model = SimpleCNN(num_classes=10)
        save_run_model(model, val_accuracy=85.0, signature=mock_signature)
        mock_mlflow.pytorch.log_model.assert_called_once_with(model, name="best_epoch_model", signature=mock_signature)


class TestSaveBestOverallModel:
    @patch("src.experiment.model_saver.mlflow")
    def test_logs_model_and_metrics(self, mock_mlflow, mock_signature):
        """save_best_overall_model should log the model, val accuracy, and run index."""
        model = SimpleCNN(num_classes=10)
        save_best_overall_model(model, val_accuracy=92.0, run_index=2, num_runs=5, signature=mock_signature)

        mock_mlflow.pytorch.log_model.assert_called_once_with(
            model, name="best_model_across_runs", signature=mock_signature
        )
        mock_mlflow.log_metric.assert_any_call("best_run_val_accuracy", 92.0)
        mock_mlflow.log_param.assert_any_call("best_run_index", 2)
