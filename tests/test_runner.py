from unittest.mock import MagicMock, patch

import pytest
import torch

from src.experiment.runner import ExperimentRunner
from src.models.simple_cnn import SimpleCNN


@pytest.fixture
def mock_registries():
    """Model and optimizer registries for runner tests."""
    model_registry = {"SimpleCNN": SimpleCNN}
    optimizer_registry = {"SGD": torch.optim.SGD}
    return model_registry, optimizer_registry


class TestExperimentRunner:
    @patch("src.experiment.runner.mlflow")
    def test_init_loads_config(self, mock_mlflow, sample_config_path, mock_registries):
        """Runner should load and parse config on init and set up MLflow experiment."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)
        assert runner.config.model.name == "SimpleCNN"
        mock_mlflow.set_experiment.assert_called_once()

    @patch("src.experiment.runner.mlflow")
    def test_get_model(self, mock_mlflow, sample_config_path, mock_registries):
        """_get_model should return an instance of the configured model class."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)
        model = runner._get_model()
        assert isinstance(model, SimpleCNN)

    @patch("src.experiment.runner.mlflow")
    def test_get_criterion(self, mock_mlflow, sample_config_path, mock_registries):
        """_get_criterion should return the loss function specified in config."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)
        criterion = runner._get_criterion()
        assert isinstance(criterion, torch.nn.CrossEntropyLoss)

    @patch("src.experiment.runner.mlflow")
    def test_get_dataloaders_missing_raises(self, mock_mlflow, sample_config_path, mock_registries):
        """_get_dataloaders with a missing data directory should raise FileNotFoundError."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)
        with pytest.raises(FileNotFoundError):
            runner._get_dataloaders("no_noise")

    @patch("src.experiment.runner.mlflow")
    def test_setup_mlflow_formats_name(self, mock_mlflow, sample_config_path, mock_registries):
        """MLflow experiment name should be formatted with model and dataset names."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)
        mock_mlflow.set_experiment.assert_called_with("test_SimpleCNN_TestDataset")

    @patch("src.experiment.runner.save_summary_to_csv")
    @patch("src.experiment.runner.generate_summary_table")
    @patch("src.experiment.runner.get_dataloaders")
    @patch("src.experiment.runner.train_single_run")
    @patch("src.experiment.runner.evaluate_model")
    @patch("src.experiment.runner.mlflow")
    def test_run_single_experiment(
        self,
        mock_mlflow,
        mock_eval,
        mock_train,
        mock_dataloaders,
        mock_summary_table,
        mock_csv,
        sample_config_path,
        mock_registries,
    ):
        """Single experiment run should call train_single_run and return a result dict."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)

        from src.training.single_run import SingleRunResult

        mock_result = SingleRunResult(
            epoch_losses=[1.0],
            val_metrics_history=[{"accuracy": 50.0}],
            best_val_metrics={"accuracy": 50.0},
            best_model_state=SimpleCNN(num_classes=10).state_dict(),
            best_val_accuracy=50.0,
            best_epoch=1,
            convergence_time=None,
            stopped_epoch=-1,
        )
        mock_train.return_value = mock_result
        mock_eval.return_value = {"accuracy": 50.0, "f1_score": 45.0}

        import numpy as np
        from mlflow.models import ModelSignature

        mock_loader = MagicMock()
        mock_sig = MagicMock(spec=ModelSignature)

        opt_config = runner.config.grid_search.optimizers[0]
        opt_params = dict(opt_config.params)

        result = runner._run_single_experiment(
            base_run_name="SGD_no_noise",
            opt_config=opt_config,
            opt_params=opt_params,
            criterion=torch.nn.CrossEntropyLoss(),
            run_seeds=np.array([42]),
            train_loader=mock_loader,
            val_loader=mock_loader,
            test_loader=mock_loader,
            signature=mock_sig,
        )

        assert isinstance(result, dict)
        mock_train.assert_called_once()

    @patch("src.experiment.runner.save_summary_to_csv")
    @patch("src.experiment.runner.generate_summary_table")
    @patch("src.experiment.runner.get_dataloaders")
    @patch("src.experiment.runner.train_single_run")
    @patch("src.experiment.runner.evaluate_model")
    @patch("src.experiment.runner.mlflow")
    def test_run_single_experiment_save_mode_all(
        self,
        mock_mlflow,
        mock_eval,
        mock_train,
        mock_dataloaders,
        mock_summary_table,
        mock_csv,
        sample_config_path,
        mock_registries,
        sample_config_dict,
        tmp_path,
    ):
        """save_model_mode='all' should call save_run_model for each run."""
        import yaml

        sample_config_dict["training"]["save_model_mode"] = "all"
        config_file = tmp_path / "cfg_all.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(str(config_file), model_reg, opt_reg)

        from src.training.single_run import SingleRunResult

        mock_result = SingleRunResult(
            epoch_losses=[1.0],
            val_metrics_history=[{"accuracy": 50.0}],
            best_val_metrics={"accuracy": 50.0},
            best_model_state=SimpleCNN(num_classes=10).state_dict(),
            best_val_accuracy=50.0,
            best_epoch=1,
            convergence_time=None,
            stopped_epoch=-1,
        )
        mock_train.return_value = mock_result
        mock_eval.return_value = {"accuracy": 50.0, "f1_score": 45.0}

        import numpy as np
        from mlflow.models import ModelSignature

        mock_loader = MagicMock()
        mock_sig = MagicMock(spec=ModelSignature)

        with patch("src.experiment.runner.save_run_model") as mock_save:
            result = runner._run_single_experiment(
                base_run_name="SGD_no_noise",
                opt_config=runner.config.grid_search.optimizers[0],
                opt_params=dict(runner.config.grid_search.optimizers[0].params),
                criterion=torch.nn.CrossEntropyLoss(),
                run_seeds=np.array([42]),
                train_loader=mock_loader,
                val_loader=mock_loader,
                test_loader=mock_loader,
                signature=mock_sig,
            )
            mock_save.assert_called_once()

    @patch("src.experiment.runner.save_best_overall_model")
    @patch("src.experiment.runner.print_aggregated_summary")
    @patch("src.experiment.runner.log_aggregated_to_parent_run")
    @patch("src.experiment.runner.log_child_run_summary")
    @patch("src.experiment.runner.log_epoch_metrics")
    @patch("src.experiment.runner.log_parent_run_params")
    @patch("src.experiment.runner.save_summary_to_csv")
    @patch("src.experiment.runner.generate_summary_table")
    @patch("src.experiment.runner.get_dataloaders")
    @patch("src.experiment.runner.train_single_run")
    @patch("src.experiment.runner.evaluate_model")
    @patch("src.experiment.runner.mlflow")
    def test_run_single_experiment_save_mode_best(
        self,
        mock_mlflow,
        mock_eval,
        mock_train,
        mock_dataloaders,
        mock_summary_table,
        mock_csv,
        mock_log_parent,
        mock_log_epoch,
        mock_log_child,
        mock_log_agg,
        mock_print_agg,
        mock_save_best,
        sample_config_dict,
        mock_registries,
        tmp_path,
    ):
        """save_model_mode='best' with multiple runs should call save_best_overall_model once."""
        import yaml

        sample_config_dict["training"]["save_model_mode"] = "best"
        sample_config_dict["training"]["num_runs"] = 2
        config_file = tmp_path / "cfg_best.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(str(config_file), model_reg, opt_reg)

        from src.training.single_run import SingleRunResult

        mock_result = SingleRunResult(
            epoch_losses=[1.0],
            val_metrics_history=[{"accuracy": 50.0}],
            best_val_metrics={"accuracy": 50.0},
            best_model_state=SimpleCNN(num_classes=10).state_dict(),
            best_val_accuracy=50.0,
            best_epoch=1,
            convergence_time=5.0,
            stopped_epoch=-1,
        )
        mock_train.return_value = mock_result
        mock_eval.return_value = {"accuracy": 50.0, "f1_score": 45.0}

        import numpy as np

        mock_loader = MagicMock()
        mock_sig = MagicMock()

        result = runner._run_single_experiment(
            base_run_name="SGD_no_noise",
            opt_config=runner.config.grid_search.optimizers[0],
            opt_params=dict(runner.config.grid_search.optimizers[0].params),
            criterion=torch.nn.CrossEntropyLoss(),
            run_seeds=np.array([42, 43]),
            train_loader=mock_loader,
            val_loader=mock_loader,
            test_loader=mock_loader,
            signature=mock_sig,
        )
        mock_save_best.assert_called_once()

    @patch("src.experiment.runner.save_summary_to_csv")
    @patch("src.experiment.runner.generate_summary_table")
    @patch("src.experiment.runner.infer_signature")
    @patch("src.experiment.runner.get_dataloaders")
    @patch("src.experiment.runner.train_single_run")
    @patch("src.experiment.runner.evaluate_model")
    @patch("src.experiment.runner.mlflow")
    def test_run_full_without_tuner(
        self,
        mock_mlflow,
        mock_eval,
        mock_train,
        mock_dataloaders,
        mock_infer_sig,
        mock_summary_table,
        mock_csv,
        sample_config_path,
        mock_registries,
    ):
        """Full run without tuner should train once per (optimizer, scenario) combination."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)

        from src.training.single_run import SingleRunResult

        mock_result = SingleRunResult(
            epoch_losses=[1.0],
            val_metrics_history=[{"accuracy": 50.0}],
            best_val_metrics={"accuracy": 50.0},
            best_model_state=None,
            best_val_accuracy=50.0,
            best_epoch=1,
            convergence_time=2.0,
            stopped_epoch=-1,
        )
        mock_train.return_value = mock_result
        mock_eval.return_value = {"accuracy": 50.0, "f1_score": 45.0}

        mock_batch = (torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,)))
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(side_effect=lambda: iter([mock_batch]))
        mock_dataloaders.return_value = (mock_loader, mock_loader, mock_loader)
        mock_infer_sig.return_value = MagicMock()

        runner.run()

        assert mock_train.call_count == 2
        mock_summary_table.assert_called_once()
        mock_csv.assert_called_once()

    @patch("src.experiment.runner.save_summary_to_csv")
    @patch("src.experiment.runner.generate_summary_table")
    @patch("src.experiment.runner.infer_signature")
    @patch("src.experiment.runner.get_dataloaders")
    @patch("src.experiment.runner.HyperparameterTuner")
    @patch("src.experiment.runner.train_single_run")
    @patch("src.experiment.runner.evaluate_model")
    @patch("src.experiment.runner.mlflow")
    def test_run_full_with_tuner(
        self,
        mock_mlflow,
        mock_eval,
        mock_train,
        mock_tuner_cls,
        mock_dataloaders,
        mock_infer_sig,
        mock_summary_table,
        mock_csv,
        sample_config_dict,
        mock_registries,
        tmp_path,
    ):
        """Full run with tuner enabled should invoke the tuner for each tunable scenario."""
        import yaml

        sample_config_dict["training"]["use_tuner"] = True
        sample_config_dict["training"]["tuner"] = {
            "n_trials": 1,
            "timeout": 10,
            "tune_scenarios": ["no_noise", "gaussian_0.05"],
        }
        config_file = tmp_path / "cfg_tuner.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(str(config_file), model_reg, opt_reg)

        mock_tuner_instance = MagicMock()
        mock_tuner_instance.tune.return_value = {"lr": 0.05, "momentum": 0.9, "weight_decay": 0.001}
        mock_tuner_cls.return_value = mock_tuner_instance

        from src.training.single_run import SingleRunResult

        mock_result = SingleRunResult(
            epoch_losses=[1.0],
            val_metrics_history=[{"accuracy": 50.0}],
            best_val_metrics={"accuracy": 50.0},
            best_model_state=None,
            best_val_accuracy=50.0,
            best_epoch=1,
            convergence_time=None,
            stopped_epoch=-1,
        )
        mock_train.return_value = mock_result
        mock_eval.return_value = {"accuracy": 50.0, "f1_score": 45.0}

        mock_batch = (torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,)))
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(side_effect=lambda: iter([mock_batch]))
        mock_dataloaders.return_value = (mock_loader, mock_loader, mock_loader)
        mock_infer_sig.return_value = MagicMock()

        runner.run()

        assert mock_tuner_cls.call_count == 2
        assert mock_tuner_instance.tune.call_count == 2

    @patch("src.experiment.runner.save_summary_to_csv")
    @patch("src.experiment.runner.generate_summary_table")
    @patch("src.experiment.runner.infer_signature")
    @patch("src.experiment.runner.get_dataloaders")
    @patch("src.experiment.runner.train_single_run")
    @patch("src.experiment.runner.evaluate_model")
    @patch("src.experiment.runner.mlflow")
    def test_run_with_convergence_and_no_convergence(
        self,
        mock_mlflow,
        mock_eval,
        mock_train,
        mock_dataloaders,
        mock_infer_sig,
        mock_summary_table,
        mock_csv,
        sample_config_path,
        mock_registries,
    ):
        """Mixed convergence results should show 'N/A' for non-converged experiments in summary."""
        model_reg, opt_reg = mock_registries
        runner = ExperimentRunner(sample_config_path, model_reg, opt_reg)

        from src.training.single_run import SingleRunResult

        call_count = [0]

        def make_result(*args, **kwargs):
            call_count[0] += 1
            return SingleRunResult(
                epoch_losses=[1.0],
                val_metrics_history=[{"accuracy": 50.0}],
                best_val_metrics={"accuracy": 50.0},
                best_model_state=None,
                best_val_accuracy=50.0,
                best_epoch=1,
                convergence_time=5.0 if call_count[0] == 1 else None,
                stopped_epoch=-1,
            )

        mock_train.side_effect = make_result
        mock_eval.return_value = {"accuracy": 50.0, "f1_score": 45.0}

        mock_batch = (torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,)))
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(side_effect=lambda: iter([mock_batch]))
        mock_dataloaders.return_value = (mock_loader, mock_loader, mock_loader)
        mock_infer_sig.return_value = MagicMock()

        runner.run()

        mock_summary_table.assert_called_once()
        summary_data = mock_summary_table.call_args[0][0]
        assert len(summary_data) == 2
        assert "N/A" in summary_data[1]["Conv. Time, s"]
