import copy
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow.models import infer_signature

from src.config import ExperimentConfig, OptimizerConfig, load_config
from src.data.processing import get_dataloaders
from src.experiment.metrics import calculate_aggregated_metrics, generate_summary_table, save_summary_to_csv
from src.experiment.mlflow_logger import (
    log_aggregated_to_parent_run,
    log_child_run_summary,
    log_epoch_metrics,
    log_parent_run_params,
    print_aggregated_summary,
)
from src.experiment.model_saver import save_best_overall_model, save_run_model
from src.training.evaluate import evaluate_model
from src.training.single_run import train_single_run
from src.utils import SPLIT_RANDOM_STATE, set_random_seed


class ExperimentRunner:
    """Orchestrates the full grid-search experiment across optimizers and noise scenarios."""

    def __init__(
        self,
        config_path: str,
        model_registry: dict[str, type],
        noise_registry: dict[str, type],
        optimizer_registry: dict[str, type],
    ) -> None:
        """
        Initializes ExperimentRunner.

        Args:
            config_path: Path to the YAML configuration file.
            model_registry: Dictionary of available model classes.
            noise_registry: Dictionary of available noise/transformation classes.
            optimizer_registry: Dictionary of available optimizer classes.
        """
        self.config: ExperimentConfig = load_config(config_path)
        self.model_registry = model_registry
        self.noise_registry = noise_registry
        self.optimizer_registry = optimizer_registry
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Sets up the MLflow experiment based on the config."""
        experiment_name = self.config.mlflow.experiment_name.format(
            model_name=self.config.model.name,
            dataset_name=self.config.data.dataset_name,
        )
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: '{experiment_name}'")

    def _get_model(self) -> nn.Module:
        """Creates and returns a model instance according to the config."""
        return self.model_registry[self.config.model.name](**self.config.model.params).to(self.device)

    def _get_criterion(self) -> nn.Module:
        """Creates and returns a loss function according to the config."""
        return getattr(nn, self.config.training.criterion)()

    def _run_single_experiment(
        self,
        base_run_name: str,
        scenario_name: str,
        opt_config: OptimizerConfig,
        criterion: nn.Module,
        run_seeds: np.ndarray,
    ) -> dict[str, Any]:
        """Runs one (optimizer, scenario) combination across all seeds."""
        training = self.config.training
        num_runs = training.num_runs

        print(f"\n--- Running {base_run_name} ({num_runs} times) ---")

        train_loader, val_loader, test_loader = get_dataloaders(
            preprocessed_root_path=self.config.data.preprocessed_root_path,
            scenario_folder_template=self.config.data.scenario_folder_template,
            scenario_name=scenario_name,
            random_state=SPLIT_RANDOM_STATE,
            batch_size=training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            subset_size=self.config.data.debug_subset_size,
        )

        input_example = next(iter(train_loader))[0][:1].cpu().numpy()
        signature = infer_signature(input_example)

        run_results_list = []
        best_overall_accuracy = -1.0
        best_overall_model_state = None
        best_run_index = -1

        with mlflow.start_run(run_name=base_run_name):
            log_parent_run_params(self.config.model, opt_config, num_runs)

            for i, run_seed in enumerate(run_seeds):
                with mlflow.start_run(run_name=f"{base_run_name}_run_{i + 1}", nested=True):
                    mlflow.log_param("random_state", run_seed)
                    set_random_seed(int(run_seed))

                    model = self._get_model()
                    optimizer = self.optimizer_registry[opt_config.name](
                        model.parameters(), lr=training.learning_rate, **opt_config.params
                    )

                    result = train_single_run(
                        model, optimizer, criterion, train_loader, val_loader, training, self.device
                    )

                    log_epoch_metrics(result)

                    if result.best_model_state is not None:
                        model.load_state_dict(result.best_model_state)

                    test_metrics = evaluate_model(model, test_loader, self.device)
                    log_child_run_summary(result, test_metrics)

                    if training.save_model_mode == "all":
                        save_run_model(model, result.best_val_accuracy, signature)

                    if training.save_model_mode == "best" and result.best_val_accuracy > best_overall_accuracy:
                        print(f"    - A new best launch has been found (Val Acc: {result.best_val_accuracy:.2f}%)")
                        best_overall_accuracy = result.best_val_accuracy
                        best_run_index = i + 1
                        best_overall_model_state = copy.deepcopy(model.state_dict())

                run_results_list.append(
                    {
                        "metrics": test_metrics,
                        "time_metric": result.convergence_time,
                        "best_val_metrics": result.best_val_metrics,
                        "val_metrics_history": result.val_metrics_history,
                    }
                )

            if training.save_model_mode == "best" and best_overall_model_state is not None:
                best_model = self._get_model()
                best_model.load_state_dict(best_overall_model_state)
                save_best_overall_model(best_model, best_overall_accuracy, best_run_index, num_runs, signature)

            if not run_results_list:
                print("Warning: No runs were performed, aggregation is not possible.")
                return {}

            aggregated_metrics = calculate_aggregated_metrics(run_results_list)
            log_aggregated_to_parent_run(aggregated_metrics)
            print_aggregated_summary(aggregated_metrics, base_run_name)

        return aggregated_metrics

    def run(self) -> None:
        """Launches the full grid of experiments."""
        print(f"The experiment is running on the device: {self.device}")

        criterion = self._get_criterion()
        run_seeds = np.random.randint(0, 2**32 - 1, size=self.config.training.num_runs)
        print(f"\nGenerated seeds: {run_seeds}\n")

        summary_data_full = []
        for scenario_name in self.config.grid_search.noise_scenarios:
            print(f"\n{'=' * 80}\nSCENARIO: {scenario_name}\n{'=' * 80}")
            for opt_config in self.config.grid_search.optimizers:
                base_run_name = f"{opt_config.name}_{scenario_name}"
                run_results = self._run_single_experiment(
                    base_run_name, scenario_name, opt_config, criterion, run_seeds
                )
                summary_data_full.append(
                    {
                        "experiment": base_run_name,
                        "hyperparams": str(opt_config.params),
                        "epochs_num": self.config.training.epochs,
                        "mean_time_s": run_results.get("mean_time_s", 0),
                        "time_std_s": run_results.get("time_std_s", 0),
                        "full_metrics": run_results.get("test_metrics", {}),
                    }
                )

        summary_data_console = []
        for row in summary_data_full:
            console_row = {
                "Experiment": row["experiment"],
                "Conv. Time, s": f"{row['mean_time_s']:.2f} ± {row['time_std_s']:.2f}",
            }
            for name, data in row["full_metrics"].items():
                console_row[f"{name.capitalize()}, %"] = f"{data.get('mean', 0):.2f} ± {data.get('std', 0):.2f}"
            summary_data_console.append(console_row)

        generate_summary_table(summary_data_console)
        save_summary_to_csv(summary_data_full)
        print("\nAll experiments were completed successfully.")
