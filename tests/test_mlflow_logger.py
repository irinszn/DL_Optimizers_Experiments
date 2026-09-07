from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelConfig, OptimizerConfig
from src.experiment.mlflow_logger import (
    find_best_model_uri,
    log_aggregated_to_parent_run,
    log_child_run_summary,
    log_epoch_metrics,
    log_parent_run_params,
    print_aggregated_summary,
)
from src.training.single_run import SingleRunResult


@pytest.fixture
def sample_result():
    """A typical 3-epoch training result for logger tests."""
    return SingleRunResult(
        epoch_losses=[2.0, 1.5, 1.0],
        val_metrics_history=[
            {"accuracy": 40.0, "f1_score": 35.0},
            {"accuracy": 50.0, "f1_score": 45.0},
            {"accuracy": 55.0, "f1_score": 50.0},
        ],
        best_val_metrics={"accuracy": 55.0, "f1_score": 50.0},
        best_model_state=None,
        best_val_accuracy=55.0,
        best_epoch=3,
        convergence_time=5.0,
        stopped_epoch=-1,
    )


class TestLogParentRunParams:
    @patch("src.experiment.mlflow_logger.mlflow")
    def test_logs_params(self, mock_mlflow):
        """Should log model params, optimizer params, optimizer name, and num_runs."""
        model_cfg = ModelConfig(name="SimpleCNN", params={"num_classes": 10})
        opt_cfg = OptimizerConfig(name="SGD", params={"lr": 0.01, "momentum": 0.9})
        log_parent_run_params(model_cfg, opt_cfg, num_runs=3)

        assert mock_mlflow.log_params.call_count == 2
        mock_mlflow.log_param.assert_any_call("optimizer_name", "SGD")
        mock_mlflow.log_param.assert_any_call("num_runs", 3)


class TestLogEpochMetrics:
    @patch("src.experiment.mlflow_logger.mlflow")
    def test_logs_losses_and_val_metrics(self, mock_mlflow, sample_result):
        """Should log epoch_loss and val_accuracy for each epoch with step numbers."""
        log_epoch_metrics(sample_result)

        loss_calls = [c for c in mock_mlflow.log_metric.call_args_list if c[0][0] == "epoch_loss"]
        assert len(loss_calls) == 3

        val_acc_calls = [c for c in mock_mlflow.log_metric.call_args_list if c[0][0] == "val_accuracy"]
        assert len(val_acc_calls) == 3

    @patch("src.experiment.mlflow_logger.mlflow")
    def test_logs_stopped_epoch(self, mock_mlflow):
        """Positive stopped_epoch should be logged as a metric."""
        result = SingleRunResult(
            epoch_losses=[1.0],
            val_metrics_history=[{"accuracy": 50.0}],
            best_val_metrics={"accuracy": 50.0},
            best_model_state=None,
            best_val_accuracy=50.0,
            best_epoch=1,
            convergence_time=None,
            stopped_epoch=5,
        )
        log_epoch_metrics(result)
        mock_mlflow.log_metric.assert_any_call("stopped_epoch", 5)

    @patch("src.experiment.mlflow_logger.mlflow")
    def test_no_stopped_epoch_when_negative(self, mock_mlflow, sample_result):
        """Negative stopped_epoch (-1) should not be logged."""
        log_epoch_metrics(sample_result)
        stopped_calls = [c for c in mock_mlflow.log_metric.call_args_list if c[0][0] == "stopped_epoch"]
        assert len(stopped_calls) == 0


class TestLogChildRunSummary:
    @patch("src.experiment.mlflow_logger.mlflow")
    def test_logs_summary(self, mock_mlflow, sample_result):
        """Should log convergence_time, best_epoch, best_val_accuracy, and test metrics."""
        test_metrics = {"accuracy": 60.0, "f1_score": 55.0}
        log_child_run_summary(sample_result, test_metrics)

        mock_mlflow.log_metric.assert_any_call("convergence_time", 5.0)
        mock_mlflow.log_metric.assert_any_call("best_epoch", 3)
        mock_mlflow.log_metric.assert_any_call("best_val_accuracy", 55.0)
        mock_mlflow.log_metrics.assert_called_once()

    @patch("src.experiment.mlflow_logger.mlflow")
    def test_no_convergence_time(self, mock_mlflow):
        """None convergence_time should not be logged."""
        result = SingleRunResult(
            epoch_losses=[1.0],
            val_metrics_history=[{"accuracy": 50.0}],
            best_val_metrics={"accuracy": 50.0},
            best_model_state=None,
            best_val_accuracy=50.0,
            best_epoch=1,
            convergence_time=None,
            stopped_epoch=-1,
        )
        log_child_run_summary(result, {"accuracy": 50.0})
        conv_calls = [c for c in mock_mlflow.log_metric.call_args_list if c[0][0] == "convergence_time"]
        assert len(conv_calls) == 0


class TestLogAggregatedToParentRun:
    @patch("src.experiment.mlflow_logger.mlflow")
    def test_logs_aggregated(self, mock_mlflow):
        """Should log converged_runs, mean_convergence_time, and per-metric aggregates."""
        agg = {
            "converged_runs": 2,
            "mean_time_s": 10.0,
            "time_std_s": 1.5,
            "test_metrics": {
                "accuracy": {"mean": 90.0, "std": 1.0, "ci_95_lower": 88.0, "ci_95_upper": 92.0},
            },
            "validation_metrics": {
                "accuracy": {"mean": 88.0, "std": 0.5},
            },
        }
        log_aggregated_to_parent_run(agg)

        mock_mlflow.log_metric.assert_any_call("converged_runs", 2)
        mock_mlflow.log_metric.assert_any_call("mean_convergence_time", 10.0)
        mock_mlflow.log_metric.assert_any_call("test_accuracy_mean", 90.0)
        mock_mlflow.log_metric.assert_any_call("val_accuracy_mean", 88.0)

    @patch("src.experiment.mlflow_logger.mlflow")
    def test_no_convergence(self, mock_mlflow):
        """Zero converged runs with None mean_time_s should still log converged_runs=0."""
        agg = {"converged_runs": 0, "mean_time_s": None, "test_metrics": {}, "validation_metrics": {}}
        log_aggregated_to_parent_run(agg)
        mock_mlflow.log_metric.assert_any_call("converged_runs", 0)


class TestFindBestModelUri:
    @patch("src.experiment.mlflow_logger.MlflowClient")
    def test_no_experiment(self, mock_client_cls):
        """Missing experiment should return None."""
        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = None
        result = find_best_model_uri("exp", "SGD", "no_noise")
        assert result is None

    @patch("src.experiment.mlflow_logger.MlflowClient")
    def test_no_runs(self, mock_client_cls):
        """Experiment with no matching runs should return None."""
        client = mock_client_cls.return_value
        exp = MagicMock()
        exp.experiment_id = "1"
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = []
        result = find_best_model_uri("exp", "SGD", "no_noise")
        assert result is None

    @patch("src.experiment.mlflow_logger.MlflowClient")
    def test_best_mode_parent_run(self, mock_client_cls):
        """Parent run with best_run_val_accuracy should return best_model_across_runs URI."""
        client = mock_client_cls.return_value
        exp = MagicMock()
        exp.experiment_id = "1"
        client.get_experiment_by_name.return_value = exp

        run = MagicMock()
        run.info.run_id = "abc123"
        run.data.metrics = {"best_run_val_accuracy": 95.0}
        client.search_runs.return_value = [run]

        result = find_best_model_uri("exp", "SGD", "no_noise")
        assert result == "runs:/abc123/best_model_across_runs"

    @patch("src.experiment.mlflow_logger.MlflowClient")
    def test_all_mode_child_run(self, mock_client_cls):
        """Parent without best_run_val_accuracy should fall back to best child run URI."""
        client = mock_client_cls.return_value
        exp = MagicMock()
        exp.experiment_id = "1"
        client.get_experiment_by_name.return_value = exp

        parent_run = MagicMock()
        parent_run.info.run_id = "parent1"
        parent_run.data.metrics = {}

        child_run = MagicMock()
        child_run.info.run_id = "child1"
        child_run.data.metrics = {"final_val_accuracy": 90.0}

        client.search_runs.side_effect = [[parent_run], [child_run]]

        result = find_best_model_uri("exp", "SGD", "no_noise")
        assert result == "runs:/child1/final_model"


class TestPrintAggregatedSummary:
    def test_empty_metrics(self):
        """Empty aggregated dict should print without errors."""
        print_aggregated_summary({}, "test_run")

    def test_full_metrics(self):
        """Full aggregated dict with all fields should print without errors."""
        agg = {
            "mean_time_s": 10.0,
            "time_std_s": 1.0,
            "converged_runs": 3,
            "runs_count": 3,
            "validation_metrics": {
                "accuracy": {"mean": 88.0, "std": 1.0},
                "f1_score": {"mean": 85.0, "std": 1.5},
            },
            "test_metrics": {
                "accuracy": {"mean": 90.0, "std": 0.5},
                "f1_score": {"mean": 87.0, "std": 0.8},
            },
        }
        print_aggregated_summary(agg, "SGD_no_noise")

    def test_no_convergence(self):
        """Aggregated dict with no convergence should print without errors."""
        agg = {
            "mean_time_s": None,
            "converged_runs": 0,
            "runs_count": 3,
            "validation_metrics": {"accuracy": {"mean": 50.0, "std": 5.0}, "f1_score": {"mean": 45.0, "std": 4.0}},
            "test_metrics": {"accuracy": {"mean": 48.0, "std": 6.0}, "f1_score": {"mean": 43.0, "std": 5.0}},
        }
        print_aggregated_summary(agg, "SGD_no_noise")
