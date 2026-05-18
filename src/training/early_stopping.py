import copy
from typing import Optional

import torch.nn as nn


class EarlyStopping:
    """Tracks best model state and triggers early stopping when patience runs out."""

    def __init__(self, patience: int, metric: str = "accuracy") -> None:
        self.patience = patience
        self.metric = metric
        self.epochs_without_improvement = 0
        self.best_val_accuracy: float = -1.0
        self.best_epoch: int = 0
        self.best_model_state: dict | None = None
        self.best_val_metrics: dict[str, float] = {}
        self.stopped_epoch: int = -1

    def step(self, val_metrics: dict[str, float], model: nn.Module, epoch: int) -> bool:
        """
        Updates internal state based on current epoch results.

        Args:
            val_metrics: Validation metrics for the current epoch.
            model: Current model — state dict is deep-copied on improvement.
            epoch: Current epoch index (0-based).

        Returns:
            True if training should stop, False otherwise.
        """
        current_accuracy = val_metrics.get(self.metric, 0.0)

        if current_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_accuracy
            self.best_epoch = epoch + 1
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_val_metrics = val_metrics
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        if self.patience > 0 and self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = epoch + 1
            print(f"    - Early stopping at epoch {epoch + 1} (no improvement for {self.patience} epochs)")
            return True

        return False
