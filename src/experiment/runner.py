import copy
import time
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from mlflow.models import infer_signature

from src.data.processing import get_dataloaders_from_drive
from src.experiment.metrics import calculate_aggregated_metrics, generate_summary_table, save_summary_to_csv
from src.training.evaluate import evaluate_model
from src.training.train import train_one_epoch
from src.utils import set_random_seed


class ExperimentRunner:
    """A class for managing, running, and logging experiments based on a given configuration."""

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
            noise_registry: Dictionary with available noise/transformation classes.
            optimizer_registry: Dictionary of available optimizer classes.
        """
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_registry = model_registry
        self.noise_registry = noise_registry
        self.optimizer_registry = optimizer_registry
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Sets up an experiment in MLflow."""
        exp_name_template = self.config["mlflow"]["experiment_name"]
        experiment_name = exp_name_template.format(
            model_name=self.config["model"]["name"],
            dataset_name=self.config["data"]["dataset_name"],
        )
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: '{experiment_name}'")

    def _get_model(self) -> nn.Module:
        """Creates and returns an instance of the model according to the config."""
        model_name = self.config["model"]["name"]
        model_params = self.config["model"].get("params", {})
        return self.model_registry[model_name](**model_params).to(self.device)

    def _get_criterion(self) -> nn.Module:
        """Creates and returns a loss function."""
        return getattr(nn, self.config["training"]["criterion"])()

    def _run_single_experiment(
        self,
        base_run_name: str,
        scenario_name: str,
        opt_config: dict,
        criterion: nn.Module,
        run_seeds: np.ndarray,
    ) -> dict[str, Any]:
        """Performs one full experiment (optimizer + scenario) with N runs."""
        run_results_list = []
        training_params = self.config["training"]
        save_mode = training_params.get("save_model_mode", "best")
        num_runs = training_params.get("num_runs", 1)
        opt_name = opt_config["name"]
        target_loss = training_params.get("target_loss", 0.3)

        print(f"\n--- Running {base_run_name} ({num_runs} times) ---")

        with mlflow.start_run(run_name=base_run_name) as parent_run:
            mlflow.log_params(self.config["model"]["params"])
            mlflow.log_params(opt_config["params"])
            mlflow.log_param("optimizer_name", opt_name)
            mlflow.log_param("num_runs", num_runs)

            best_overall_run_accuracy = -1.0
            best_overall_model_state = None
            best_run_index = -1

            train_loader, val_loader, test_loader = get_dataloaders_from_drive(
                preprocessed_root_path=self.config["data"]["preprocessed_root_path"],
                scenario_folder_template=self.config["data"]["scenario_folder_template"],
                scenario_name=scenario_name,
                random_state=42,
                batch_size=self.config["training"]["batch_size"],
                subset_size=self.config["data"].get("debug_subset_size"),
            )

            input_example = next(iter(train_loader))[0][:1].cpu().numpy()
            signature = infer_signature(input_example)

            for i, run_seed in enumerate(run_seeds):
                with mlflow.start_run(run_name=f"{base_run_name}_run_{i + 1}", nested=True):
                    mlflow.log_param("random_state", run_seed)
                    set_random_seed(int(run_seed))

                    model = self._get_model()
                    optimizer_class = self.optimizer_registry[opt_name]
                    optimizer = optimizer_class(
                        model.parameters(),
                        lr=training_params["learning_rate"],
                        **opt_config["params"],
                    )

                    val_metrics_history = []
                    convergence_time: float = -1.0
                    start_time = time.time()

                    for epoch in range(training_params["epochs"]):
                        epoch_loss = train_one_epoch(model, optimizer, criterion, train_loader, self.device)
                        val_metrics = evaluate_model(model, val_loader, self.device)
                        val_metrics_history.append(val_metrics)

                        if epoch_loss <= target_loss and convergence_time < 0:
                            convergence_time = time.time() - start_time

                        mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
                        for name, value in val_metrics.items():
                            mlflow.log_metric(f"val_{name}", value, step=epoch)

                    total_time = time.time() - start_time
                    time_to_log = convergence_time if convergence_time > 0 else total_time
                    mlflow.log_metric("convergence_time", time_to_log)

                    test_metrics = evaluate_model(model, test_loader, self.device)
                    run_results_list.append(
                        {
                            "metrics": test_metrics,
                            "time_metric": time_to_log,
                            "val_metrics_history": val_metrics_history,
                        }
                    )

                    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                    final_run_val_accuracy = val_metrics_history[-1].get("accuracy", 0) if val_metrics_history else 0
                    mlflow.log_metric("final_val_accuracy", final_run_val_accuracy)

                    if save_mode == "all":
                        print(f"    - Saving the model (Val Acc: {final_run_val_accuracy:.2f}%)")
                        mlflow.pytorch.log_model(model, name="final_model", signature=signature)

                    if save_mode == "best" and final_run_val_accuracy > best_overall_run_accuracy:
                        print(f"    - A new best launch has been found (Val Acc: {final_run_val_accuracy:.2f}%)")
                        best_overall_run_accuracy = final_run_val_accuracy
                        best_run_index = i + 1
                        best_overall_model_state = copy.deepcopy(model.state_dict())

            if save_mode == "best" and best_overall_model_state is not None:
                print(
                    f"\nSaving the BEST model from {num_runs} runs (run #{best_run_index}, Val Acc: {best_overall_run_accuracy:.2f}%)"
                )
                best_model = self._get_model()
                best_model.load_state_dict(best_overall_model_state)
                mlflow.pytorch.log_model(best_model, name="best_model_across_runs", signature=signature)
                mlflow.log_metric("best_run_val_accuracy", best_overall_run_accuracy)
                mlflow.log_param("best_run_index", best_run_index)

            if not run_results_list:
                print("Warning: No runs were performed, aggregation is not possible.")
                return {}

            aggregated_metrics = calculate_aggregated_metrics(run_results_list)

            print(f"\n--- Aggregation and logging for '{base_run_name}' ---")
            if aggregated_metrics:
                mean_time = aggregated_metrics.get("mean_time_s", 0)
                std_time = aggregated_metrics.get("time_std_s", 0)
                mlflow.log_metric("mean_time", mean_time)
                mlflow.log_metric("std_time", std_time)

                for name, data in aggregated_metrics.get("test_metrics", {}).items():
                    mlflow.log_metric(f"test_{name}_mean", data["mean"])
                    mlflow.log_metric(f"test_{name}_std", data["std"])
                    mlflow.log_metric(f"test_{name}_ci95_lower", data["ci_95_lower"])
                    mlflow.log_metric(f"test_{name}_ci95_upper", data["ci_95_upper"])

                for name, data in aggregated_metrics.get("validation_metrics", {}).items():
                    mlflow.log_metric(f"val_{name}_mean", data["mean"])
                    mlflow.log_metric(f"val_{name}_std", data["std"])

                val_metrics_agg = aggregated_metrics.get("validation_metrics", {})
                test_metrics_agg = aggregated_metrics.get("test_metrics", {})
                print(
                    f"  - Val Accuracy:  {val_metrics_agg.get('accuracy', {}).get('mean', 0):.2f} ± {val_metrics_agg.get('accuracy', {}).get('std', 0):.2f}"
                )
                print(
                    f"  - Val F1-score:  {val_metrics_agg.get('f1_score', {}).get('mean', 0):.2f} ± {val_metrics_agg.get('f1_score', {}).get('std', 0):.2f}"
                )
                print(f"  - Conv. Time, s: {mean_time:.2f} ± {std_time:.2f}")
                print(
                    f"\n  - Test Accuracy: {test_metrics_agg.get('accuracy', {}).get('mean', 0):.2f} ± {test_metrics_agg.get('accuracy', {}).get('std', 0):.2f}"
                )
                print(
                    f"  - Test F1-score: {test_metrics_agg.get('f1_score', {}).get('mean', 0):.2f} ± {test_metrics_agg.get('f1_score', {}).get('std', 0):.2f}"
                )

            return aggregated_metrics

    def run(self) -> None:
        """Launches the entire grid of experiments."""
        print(f"The experiment is running on the device: {self.device}")

        criterion = self._get_criterion()
        summary_data_full = []
        num_runs = self.config["training"].get("num_runs", 1)

        run_seeds = np.random.randint(0, 2**32 - 1, size=num_runs)
        print(f"\nGenerated seeds: {run_seeds}\n")

        for scenario_name in self.config["grid_search"]["noise_scenarios"]:
            print(f"\n{'=' * 80}\nSCENARIO: {scenario_name}\n{'=' * 80}")

            for opt_config in self.config["grid_search"]["optimizers"]:
                base_run_name = f"{opt_config['name']}_{scenario_name}"
                run_results = self._run_single_experiment(
                    base_run_name, scenario_name, opt_config, criterion, run_seeds
                )

                summary_data_full.append(
                    {
                        "experiment": base_run_name,
                        "hyperparams": str(opt_config.get("params", "default")),
                        "epochs_num": self.config["training"]["epochs"],
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
