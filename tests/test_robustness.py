from unittest.mock import MagicMock, patch

import pytest
import torch

from src.experiment.robustness import run_comparative_robustness_evaluation


class TestRunComparativeRobustnessEvaluation:
    @patch("src.experiment.robustness.get_dataloaders")
    @patch("src.experiment.robustness.evaluate_model")
    @patch("src.experiment.robustness.mlflow")
    @patch("src.experiment.robustness.find_best_model_uri")
    @patch("src.experiment.robustness.load_config")
    def test_skips_optimizer_without_model(
        self, mock_load_config, mock_find_uri, mock_mlflow, mock_eval, mock_dataloaders
    ):
        """Optimizer with no saved model (URI=None) should be skipped entirely."""
        config = MagicMock()
        config.mlflow.experiment_name = "test_{model_name}_{dataset_name}"
        config.model.name = "SimpleCNN"
        config.data.dataset_name = "TestData"
        config.robustness.trained_on_scenario = "no_noise"
        config.grid_search.noise_scenarios = {"no_noise": []}
        config.training.batch_size = 4
        config.data.preprocessed_root_path = "/tmp"
        config.data.scenario_folder_template = "Test_{scenario_name}"
        config.data.num_workers = 0
        config.data.pin_memory = False
        config.data.debug_subset_size = None
        mock_load_config.return_value = config

        mock_find_uri.return_value = None

        registry = {"SGD": torch.optim.SGD}
        run_comparative_robustness_evaluation("config.yaml", registry)

        mock_find_uri.assert_called_once()
        mock_eval.assert_not_called()

    @patch("src.experiment.robustness.pd.DataFrame.to_csv")
    @patch("src.experiment.robustness.get_dataloaders")
    @patch("src.experiment.robustness.evaluate_model")
    @patch("src.experiment.robustness.mlflow")
    @patch("src.experiment.robustness.find_best_model_uri")
    @patch("src.experiment.robustness.load_config")
    def test_evaluates_all_scenarios(
        self, mock_load_config, mock_find_uri, mock_mlflow, mock_eval, mock_dataloaders, mock_to_csv
    ):
        """All noise scenarios should be evaluated when a model URI is found."""
        config = MagicMock()
        config.mlflow.experiment_name = "test_{model_name}_{dataset_name}"
        config.model.name = "SimpleCNN"
        config.data.dataset_name = "TestData"
        config.robustness.trained_on_scenario = "no_noise"
        config.grid_search.noise_scenarios = {"no_noise": [], "gaussian_0.05": []}
        config.training.batch_size = 4
        config.data.preprocessed_root_path = "/tmp"
        config.data.scenario_folder_template = "Test_{scenario_name}"
        config.data.num_workers = 0
        config.data.pin_memory = False
        config.data.debug_subset_size = None
        mock_load_config.return_value = config

        mock_find_uri.return_value = "runs:/abc/model"

        mock_model = MagicMock()
        mock_mlflow.pytorch.load_model.return_value = mock_model

        mock_loader = MagicMock()
        mock_dataloaders.return_value = (mock_loader, mock_loader, mock_loader)

        mock_eval.return_value = {"accuracy": 90.0, "f1_score": 85.0}

        registry = {"SGD": torch.optim.SGD}
        run_comparative_robustness_evaluation("config.yaml", registry)

        assert mock_eval.call_count == 2

    @patch("src.experiment.robustness.get_dataloaders")
    @patch("src.experiment.robustness.evaluate_model")
    @patch("src.experiment.robustness.mlflow")
    @patch("src.experiment.robustness.find_best_model_uri")
    @patch("src.experiment.robustness.load_config")
    def test_handles_failed_model_load(
        self, mock_load_config, mock_find_uri, mock_mlflow, mock_eval, mock_dataloaders
    ):
        """Failed model loading should skip evaluation without crashing."""
        config = MagicMock()
        config.mlflow.experiment_name = "test_{model_name}_{dataset_name}"
        config.model.name = "SimpleCNN"
        config.data.dataset_name = "TestData"
        config.robustness.trained_on_scenario = "no_noise"
        config.grid_search.noise_scenarios = {"no_noise": []}
        config.training.batch_size = 4
        config.data.preprocessed_root_path = "/tmp"
        config.data.scenario_folder_template = "Test_{scenario_name}"
        config.data.num_workers = 0
        config.data.pin_memory = False
        config.data.debug_subset_size = None
        mock_load_config.return_value = config

        mock_find_uri.return_value = "runs:/abc/model"
        mock_mlflow.pytorch.load_model.side_effect = Exception("Model load failed")

        registry = {"SGD": torch.optim.SGD}
        run_comparative_robustness_evaluation("config.yaml", registry)

        mock_eval.assert_not_called()
