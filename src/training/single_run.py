import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.training.early_stopping import EarlyStopping
from src.training.evaluate import evaluate_model
from src.training.scheduler import build_scheduler
from src.training.train import train_one_epoch


@dataclass
class SingleRunResult:
    epoch_losses: list[float]
    val_metrics_history: list[dict[str, float]]
    best_val_metrics: dict[str, float]
    best_model_state: dict | None
    best_val_accuracy: float
    best_epoch: int
    convergence_time: float | None
    stopped_epoch: int


def train_single_run(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: TrainingConfig,
    device: torch.device,
) -> SingleRunResult:
    """Runs the full training loop with early stopping and best-model tracking."""
    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience,
        metric=training_config.early_stopping_metric,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_config=training_config.scheduler,
        total_epochs=training_config.epochs,
        base_lr=optimizer.param_groups[0]["lr"],
    )
    epoch_losses: list[float] = []
    val_metrics_history: list[dict[str, float]] = []
    convergence_time = -1.0
    start_time = time.time()

    for epoch in range(training_config.epochs):
        epoch_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_metrics = evaluate_model(model, val_loader, device)

        epoch_losses.append(epoch_loss)
        val_metrics_history.append(val_metrics)

        if epoch_loss <= training_config.target_loss and convergence_time < 0:
            convergence_time = time.time() - start_time

        if scheduler is not None:
            scheduler.step()

        if early_stopping.step(val_metrics, model, epoch):
            break

    return SingleRunResult(
        epoch_losses=epoch_losses,
        val_metrics_history=val_metrics_history,
        best_val_metrics=early_stopping.best_val_metrics,
        best_model_state=early_stopping.best_model_state,
        best_val_accuracy=early_stopping.best_val_accuracy,
        best_epoch=early_stopping.best_epoch,
        convergence_time=convergence_time if convergence_time >= 0 else None,
        stopped_epoch=early_stopping.stopped_epoch,
    )
