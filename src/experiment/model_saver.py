import mlflow
import mlflow.pytorch
import torch.nn as nn
from mlflow.models import ModelSignature


def save_run_model(model: nn.Module, val_accuracy: float, signature: ModelSignature) -> None:
    """Saves the best-epoch model for a single run (save_mode='all')."""
    print(f"    - Saving the model (Val Acc: {val_accuracy:.2f}%)")
    mlflow.pytorch.log_model(model, name="best_epoch_model", signature=signature)


def save_best_overall_model(
    model: nn.Module,
    val_accuracy: float,
    run_index: int,
    num_runs: int,
    signature: ModelSignature,
) -> None:
    """Saves the best model across all seed runs to the parent MLflow run (save_mode='best')."""
    print(f"\nSaving the BEST model from {num_runs} runs (run #{run_index}, Val Acc: {val_accuracy:.2f}%)")
    mlflow.pytorch.log_model(model, name="best_model_across_runs", signature=signature)
    mlflow.log_metric("best_run_val_accuracy", val_accuracy)
    mlflow.log_param("best_run_index", run_index)
