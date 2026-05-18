from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import ModelConfig, OptimizerConfig
from src.training.single_run import SingleRunResult


def log_parent_run_params(model_config: ModelConfig, opt_config: OptimizerConfig, num_runs: int) -> None:
    """Logs hyperparameters to the current parent MLflow run."""
    mlflow.log_params(model_config.params)
    mlflow.log_params(opt_config.params)
    mlflow.log_param("optimizer_name", opt_config.name)
    mlflow.log_param("num_runs", num_runs)


def log_epoch_metrics(result: SingleRunResult) -> None:
    """Logs per-epoch loss and validation metrics to the current child MLflow run."""
    for epoch, loss in enumerate(result.epoch_losses):
        mlflow.log_metric("epoch_loss", loss, step=epoch)
    for epoch, metrics in enumerate(result.val_metrics_history):
        for name, value in metrics.items():
            mlflow.log_metric(f"val_{name}", value, step=epoch)
    if result.stopped_epoch > 0:
        mlflow.log_metric("stopped_epoch", result.stopped_epoch)


def log_child_run_summary(result: SingleRunResult, test_metrics: dict[str, float]) -> None:
    """Logs post-training summary metrics to the current child MLflow run."""
    mlflow.log_metric("convergence_time", result.convergence_time)
    mlflow.log_metric("best_epoch", result.best_epoch)
    mlflow.log_metric("best_val_accuracy", result.best_val_accuracy)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


def log_aggregated_to_parent_run(aggregated_metrics: dict[str, Any]) -> None:
    """Logs aggregated metrics to the current parent MLflow run."""
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


def find_best_model_uri(experiment_name: str, optimizer_name: str, scenario_name: str) -> str | None:
    """
    Finds the URI of the best model by searching through the run history.

    Can work with:
    1. save_mode='best' (searches for the model in the Parent Run using the best_run_val_accuracy metric).
    2. save_mode='all' (searches for the best Child Run).
    3. Situations where the last run crashed or produced a poor result
       (the function will automatically search previous runs).
    """
    print(f"Searching for best model for '{optimizer_name}_{scenario_name}'...")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    parent_run_name = f"{optimizer_name}_{scenario_name}"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
        order_by=["attribute.start_time DESC"],
    )

    if not runs:
        print(f"No runs found with name '{parent_run_name}'.")
        return None

    print(f"  - Found {len(runs)} historical runs. Checking for valid artifacts...")

    for run in runs:
        run_id = run.info.run_id
        metrics = run.data.metrics

        if "best_run_val_accuracy" in metrics:
            acc = metrics["best_run_val_accuracy"]
            print(f"  - Found valid Parent Run (ID: {run_id}) with save_mode='best'.")
            print(f"    Validation Accuracy: {acc:.2f}%")
            return f"runs:/{run_id}/best_model_across_runs"

        child_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            order_by=["metrics.final_val_accuracy DESC"],
            max_results=1,
        )

        if child_runs:
            best_child = child_runs[0]
            child_acc = best_child.data.metrics.get("final_val_accuracy", 0)
            if child_acc > 0:
                print(f"  - Found valid Child Run (ID: {best_child.info.run_id}) in Parent ({run_id}).")
                print(f"    Validation Accuracy: {child_acc:.2f}%")
                return f"runs:/{best_child.info.run_id}/final_model"

    print(f"Could not find ANY valid model for '{parent_run_name}' in history.")
    return None


def print_aggregated_summary(aggregated_metrics: dict[str, Any], run_name: str) -> None:
    """Prints a formatted aggregation summary to console."""
    print(f"\n--- Aggregation and logging for '{run_name}' ---")
    if not aggregated_metrics:
        return

    mean_time = aggregated_metrics.get("mean_time_s", 0)
    std_time = aggregated_metrics.get("time_std_s", 0)
    val = aggregated_metrics.get("validation_metrics", {})
    test = aggregated_metrics.get("test_metrics", {})

    print(
        f"  - Val Accuracy:  {val.get('accuracy', {}).get('mean', 0):.2f} ± {val.get('accuracy', {}).get('std', 0):.2f}"
    )
    print(
        f"  - Val F1-score:  {val.get('f1_score', {}).get('mean', 0):.2f} ± {val.get('f1_score', {}).get('std', 0):.2f}"
    )
    print(f"  - Conv. Time, s: {mean_time:.2f} ± {std_time:.2f}")
    print(
        f"\n  - Test Accuracy: {test.get('accuracy', {}).get('mean', 0):.2f} ± {test.get('accuracy', {}).get('std', 0):.2f}"
    )
    print(
        f"  - Test F1-score: {test.get('f1_score', {}).get('mean', 0):.2f} ± {test.get('f1_score', {}).get('std', 0):.2f}"
    )
