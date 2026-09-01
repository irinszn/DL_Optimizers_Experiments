import copy
import logging
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow.models import ModelSignature, infer_signature
from torch.utils.data import DataLoader

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
from src.training.tuner import HyperparameterTuner
from src.types import ModelRegistry, OptimizerRegistry
from src.utils import SPLIT_RANDOM_STATE, find_nearest_tuned_scenario, set_random_seed

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full grid-search experiment across optimizers and noise scenarios."""

    def __init__(
        self,
        config_path: str,
        model_registry: ModelRegistry,
        optimizer_registry: OptimizerRegistry,
    ) -> None:
        """
        Initializes ExperimentRunner.

        Args:
            config_path: Path to the YAML configuration file.
            model_registry: Dictionary of available model classes.
            optimizer_registry: Dictionary of available optimizer classes.
        """
        self.config: ExperimentConfig = load_config(config_path)
        self.model_registry = model_registry
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
        logger.info("MLflow experiment set to: '%s'", experiment_name)

    def _get_model(self) -> nn.Module:
        """Creates and returns a model instance according to the config."""
        return self.model_registry[self.config.model.name](**self.config.model.params).to(self.device)

    def _get_criterion(self) -> nn.Module:
        """Creates and returns a loss function according to the config."""
        return getattr(nn, self.config.training.criterion)()

    def _get_dataloaders(self, scenario_name: str) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Loads dataloaders for a given scenario."""
        return get_dataloaders(
            preprocessed_root_path=self.config.data.preprocessed_root_path,
            scenario_folder_template=self.config.data.scenario_folder_template,
            scenario_name=scenario_name,
            random_state=SPLIT_RANDOM_STATE,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            subset_size=self.config.data.debug_subset_size,
        )

    def _tune_optimizers(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Runs Optuna tuning for each (optimizer, tune_scenario) pair.
        For non-tuned scenarios, assigns params from the nearest tuned scenario.

        Returns:
            Nested dict: tuned_params[optimizer_name][scenario_name] = optimizer params dict.
        """
        training = self.config.training
        tuner_cfg = training.tuner
        tune_scenarios = tuner_cfg.tune_scenarios
        all_scenarios = list(self.config.grid_search.noise_scenarios.keys())

        # {optimizer_name: {scenario_name: params_dict}}
        tuned_params: dict[str, dict[str, dict[str, Any]]] = {}

        for opt_config in self.config.grid_search.optimizers:
            tuned_params[opt_config.name] = {}
            criterion = self._get_criterion()

            for scenario_name in tune_scenarios:
                logger.info("=" * 60)
                logger.info("TUNING %s on scenario '%s'", opt_config.name, scenario_name)
                logger.info("=" * 60)

                train_loader, val_loader, _ = self._get_dataloaders(scenario_name)

                # Params in config but not in search_space are fixed (e.g. nesterov)
                tuned_keys = {"lr", "momentum", "weight_decay", "betas"}
                fixed_params = {k: v for k, v in opt_config.params.items() if k not in tuned_keys}

                tuner = HyperparameterTuner(
                    model_class=self.model_registry[self.config.model.name],
                    model_params=self.config.model.params,
                    optimizer_name=opt_config.name,
                    optimizer_factory=self.optimizer_registry[opt_config.name],
                    search_space=opt_config.search_space,
                    fixed_params=fixed_params,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    epochs_per_trial=training.epochs,
                    early_stopping_patience=training.early_stopping_patience,
                    scheduler_config=training.scheduler,
                )

                best_params = tuner.tune(
                    n_trials=tuner_cfg.n_trials,
                    timeout=tuner_cfg.timeout,
                )
                tuned_params[opt_config.name][scenario_name] = best_params
                logger.info("Best params for %s on '%s': %s", opt_config.name, scenario_name, best_params)

            # Fill non-tuned scenarios with nearest neighbor
            for scenario_name in all_scenarios:
                if scenario_name not in tuned_params[opt_config.name]:
                    nearest = find_nearest_tuned_scenario(scenario_name, tune_scenarios)
                    tuned_params[opt_config.name][scenario_name] = tuned_params[opt_config.name][nearest]
                    logger.info(
                        "Scenario '%s' for %s: using params from nearest tuned scenario '%s'",
                        scenario_name,
                        opt_config.name,
                        nearest,
                    )

        return tuned_params

    def _run_single_experiment(
        self,
        base_run_name: str,
        opt_config: OptimizerConfig,
        opt_params: dict[str, Any],
        criterion: nn.Module,
        run_seeds: np.ndarray,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        signature: ModelSignature,
    ) -> dict[str, Any]:
        """Runs one (optimizer, scenario) combination across all seeds."""
        training = self.config.training
        num_runs = training.num_runs

        logger.info("Running %s (%d times)", base_run_name, num_runs)

        run_results_list = []
        best_overall_accuracy = -1.0
        best_overall_model_state = None
        best_run_index = -1

        with mlflow.start_run(run_name=base_run_name):
            log_parent_run_params(self.config.model, opt_config, num_runs)
            mlflow.log_params({f"tuned_{k}": v for k, v in opt_params.items() if not isinstance(v, tuple)})
            for k, v in opt_params.items():
                if isinstance(v, tuple):
                    for i, val in enumerate(v):
                        mlflow.log_param(f"tuned_{k}_{i}", val)

            for i, run_seed in enumerate(run_seeds):
                with mlflow.start_run(run_name=f"{base_run_name}_run_{i + 1}", nested=True):
                    mlflow.log_param("random_state", run_seed)
                    set_random_seed(int(run_seed))

                    model = self._get_model()
                    optimizer = self.optimizer_registry[opt_config.name](model.parameters(), **opt_params)

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
                        logger.info("New best run found (Val Acc: %.2f%%)", result.best_val_accuracy)
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
                logger.warning("No runs were performed, aggregation is not possible.")
                return {}

            aggregated_metrics = calculate_aggregated_metrics(run_results_list)
            log_aggregated_to_parent_run(aggregated_metrics)
            print_aggregated_summary(aggregated_metrics, base_run_name)

        return aggregated_metrics

    def run(self) -> None:
        """Launches the full grid of experiments."""
        logger.info("The experiment is running on device: %s", self.device)

        training = self.config.training

        # Tuning phase
        tuned_params: dict[str, dict[str, dict[str, Any]]] | None = None
        if training.use_tuner:
            logger.info("=" * 80)
            logger.info("STARTING HYPERPARAMETER TUNING PHASE")
            logger.info("=" * 80)
            tuned_params = self._tune_optimizers()

        # Experiment phase
        criterion = self._get_criterion()
        run_seeds = np.random.randint(0, 2**32 - 1, size=training.num_runs)
        logger.info("Generated seeds: %s", run_seeds)

        summary_data_full = []
        for scenario_name in self.config.grid_search.noise_scenarios:
            logger.info("=" * 80)
            logger.info("SCENARIO: %s", scenario_name)
            logger.info("=" * 80)

            train_loader, val_loader, test_loader = self._get_dataloaders(scenario_name)
            signature = infer_signature(next(iter(train_loader))[0][:1].cpu().numpy())

            for opt_config in self.config.grid_search.optimizers:
                if tuned_params is not None:
                    opt_params = tuned_params[opt_config.name][scenario_name]
                else:
                    opt_params = dict(opt_config.params)

                base_run_name = f"{opt_config.name}_{scenario_name}"
                run_results = self._run_single_experiment(
                    base_run_name,
                    opt_config,
                    opt_params,
                    criterion,
                    run_seeds,
                    train_loader,
                    val_loader,
                    test_loader,
                    signature,
                )
                summary_data_full.append(
                    {
                        "experiment": base_run_name,
                        "hyperparams": str(opt_params),
                        "epochs_num": training.epochs,
                        "mean_time_s": run_results.get("mean_time_s"),
                        "time_std_s": run_results.get("time_std_s", 0),
                        "converged_runs": run_results.get("converged_runs", 0),
                        "runs_count": training.num_runs,
                        "full_metrics": run_results.get("test_metrics", {}),
                    }
                )

        summary_data_console = []
        for row in summary_data_full:
            mean_time = row["mean_time_s"]
            conv_time_str = (
                f"{mean_time:.2f} ± {row['time_std_s']:.2f} ({row['converged_runs']}/{row['runs_count']})"
                if mean_time is not None
                else f"N/A (0/{row['runs_count']})"
            )
            console_row = {
                "Experiment": row["experiment"],
                "Conv. Time, s": conv_time_str,
            }
            for name, data in row["full_metrics"].items():
                console_row[f"{name.capitalize()}, %"] = f"{data.get('mean', 0):.2f} ± {data.get('std', 0):.2f}"
            summary_data_console.append(console_row)

        generate_summary_table(summary_data_console)
        save_summary_to_csv(summary_data_full)
        logger.info("All experiments completed successfully.")
