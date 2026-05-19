import logging

import mlflow
import mlflow.pytorch
import torch.nn as nn
from mlflow.models import ModelSignature

logger = logging.getLogger(__name__)


def save_run_model(model: nn.Module, val_accuracy: float, signature: ModelSignature) -> None:
    """Saves the best-epoch model for a single run (save_mode='all')."""
    logger.info("Saving model (Val Acc: %.2f%%)", val_accuracy)
    mlflow.pytorch.log_model(model, name="best_epoch_model", signature=signature)


def save_best_overall_model(
    model: nn.Module,
    val_accuracy: float,
    run_index: int,
    num_runs: int,
    signature: ModelSignature,
) -> None:
    """Saves the best model across all seed runs to the parent MLflow run (save_mode='best')."""
    logger.info("Saving BEST model from %d runs (run #%d, Val Acc: %.2f%%)", num_runs, run_index, val_accuracy)
    mlflow.pytorch.log_model(model, name="best_model_across_runs", signature=signature)
    mlflow.log_metric("best_run_val_accuracy", val_accuracy)
    mlflow.log_param("best_run_index", run_index)
