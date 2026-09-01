import logging
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

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
    if result.convergence_time is not None:
        mlflow.log_metric("convergence_time", result.convergence_time)
    mlflow.log_metric("best_epoch", result.best_epoch)
    mlflow.log_metric("best_val_accuracy", result.best_val_accuracy)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


def log_aggregated_to_parent_run(aggregated_metrics: dict[str, Any]) -> None:
    """Logs aggregated metrics to the current parent MLflow run."""
    mlflow.log_metric("converged_runs", aggregated_metrics.get("converged_runs", 0))
    mean_time = aggregated_metrics.get("mean_time_s")
    if mean_time is not None:
        mlflow.log_metric("mean_convergence_time", mean_time)
        mlflow.log_metric("std_convergence_time", aggregated_metrics.get("time_std_s", 0))

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
    logger.info("Searching for best model for '%s_%s'...", optimizer_name, scenario_name)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning("Experiment '%s' not found.", experiment_name)
        return None

    parent_run_name = f"{optimizer_name}_{scenario_name}"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
        order_by=["attribute.start_time DESC"],
    )

    if not runs:
        logger.warning("No runs found with name '%s'.", parent_run_name)
        return None

    logger.info("Found %d historical runs. Checking for valid artifacts...", len(runs))

    for run in runs:
        run_id = run.info.run_id
        metrics = run.data.metrics

        if "best_run_val_accuracy" in metrics:
            acc = metrics["best_run_val_accuracy"]
            logger.info("Found valid Parent Run (ID: %s) with save_mode='best'. Val Acc: %.2f%%", run_id, acc)
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
                logger.info(
                    "Found valid Child Run (ID: %s) in Parent (%s). Val Acc: %.2f%%",
                    best_child.info.run_id,
                    run_id,
                    child_acc,
                )
                return f"runs:/{best_child.info.run_id}/final_model"

    logger.warning("Could not find any valid model for '%s'.", parent_run_name)
    return None


def print_aggregated_summary(aggregated_metrics: dict[str, Any], run_name: str) -> None:
    """Logs a formatted aggregation summary."""
    logger.info("--- Aggregation and logging for '%s' ---", run_name)
    if not aggregated_metrics:
        return

    mean_time = aggregated_metrics.get("mean_time_s")
    std_time = aggregated_metrics.get("time_std_s", 0)
    converged = aggregated_metrics.get("converged_runs", 0)
    total = aggregated_metrics.get("runs_count", 0)
    val = aggregated_metrics.get("validation_metrics", {})
    test = aggregated_metrics.get("test_metrics", {})

    logger.info(
        "  Val Accuracy:  %.2f ± %.2f",
        val.get("accuracy", {}).get("mean", 0),
        val.get("accuracy", {}).get("std", 0),
    )
    logger.info(
        "  Val F1-score:  %.2f ± %.2f",
        val.get("f1_score", {}).get("mean", 0),
        val.get("f1_score", {}).get("std", 0),
    )
    if mean_time is not None:
        logger.info("  Conv. Time, s: %.2f ± %.2f (%d/%d runs converged)", mean_time, std_time, converged, total)
    else:
        logger.info("  Conv. Time, s: N/A (0/%d runs reached target loss)", total)
    logger.info(
        "  Test Accuracy: %.2f ± %.2f",
        test.get("accuracy", {}).get("mean", 0),
        test.get("accuracy", {}).get("std", 0),
    )
    logger.info(
        "  Test F1-score: %.2f ± %.2f",
        test.get("f1_score", {}).get("mean", 0),
        test.get("f1_score", {}).get("std", 0),
    )
