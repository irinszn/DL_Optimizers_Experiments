import logging
from typing import Any

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import SchedulerConfig
from src.training.early_stopping import EarlyStopping
from src.training.evaluate import evaluate_model
from src.training.scheduler import build_scheduler
from src.training.train import train_one_epoch
from src.types import ModelFactory, OptimizerFactory
from src.utils import set_random_seed

logger = logging.getLogger(__name__)

TUNER_SEED = 42


class HyperparameterTuner:
    def __init__(
        self,
        model_class: ModelFactory,
        model_params: dict[str, Any],
        optimizer_name: str,
        optimizer_factory: OptimizerFactory,
        search_space: dict[str, list[float]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        fixed_params: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        epochs_per_trial: int = 12,
        early_stopping_patience: int = 3,
        scheduler_config: SchedulerConfig | None = None,
    ):
        """
        Initializes the tuner.

        Args:
            model_class: Model class.
            model_params: Parameters for the model constructor.
            optimizer_name: Name of the optimizer ('SGD', 'Adam', 'LAMB').
            optimizer_factory: Callable that creates an optimizer instance.
            search_space: Dict of param_name → [min, max] ranges for Optuna.
            train_loader: DataLoader for train data.
            val_loader: DataLoader for validation data.
            criterion: Loss function. Defaults to CrossEntropyLoss.
            epochs_per_trial: Max epochs per trial.
            early_stopping_patience: Stops a trial early if val accuracy doesn't improve.
            scheduler_config: LR scheduler configuration.
            fixed_params: Optimizer params from config that are not tuned (e.g. nesterov).
        """
        self.model_class = model_class
        self.model_params = model_params
        self.optimizer_name = optimizer_name
        self.optimizer_factory = optimizer_factory
        self.search_space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epochs_per_trial = epochs_per_trial
        self.early_stopping_patience = early_stopping_patience
        self.fixed_params = fixed_params or {}
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._suggestion_methods: dict[str, Any] = {
            "SGD": self._suggest_sgd_params,
            "Adam": self._suggest_adam_params,
            "LAMB": self._suggest_lamb_params,
        }

        if optimizer_name not in self._suggestion_methods:
            raise ValueError(
                f"Optimizer '{optimizer_name}' doesn't support. " f"Available: {list(self._suggestion_methods.keys())}"
            )

        logger.info("Tuner device: %s", self.device)
        logger.info("Tuner configured for optimizer: %s", self.optimizer_name)

    def _range(self, key: str, default_min: float, default_max: float) -> tuple[float, float]:
        """Gets search range from config or falls back to defaults."""
        r = self.search_space.get(key, [default_min, default_max])
        return r[0], r[1]

    def _suggest_sgd_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Search space for SGD."""
        lr_lo, lr_hi = self._range("lr_log10", -4, 0)
        mom_lo, mom_hi = self._range("momentum", 0.5, 0.99)
        wd_lo, wd_hi = self._range("wd_log10", -6, -3.3)
        params: dict[str, Any] = {
            "lr": 10 ** trial.suggest_float("lr_log10", lr_lo, lr_hi),
            "momentum": trial.suggest_float("momentum", mom_lo, mom_hi),
            "weight_decay": 10 ** trial.suggest_float("wd_log10", wd_lo, wd_hi),
        }
        params.update(self.fixed_params)
        return params

    def _suggest_adam_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Search space for Adam."""
        lr_lo, lr_hi = self._range("lr_log10", -5, -2)
        b1_lo, b1_hi = self._range("beta1", 0.8, 0.99)
        b2_lo, b2_hi = self._range("beta2", 0.9, 0.999)
        wd_lo, wd_hi = self._range("wd_log10", -6, -3)
        return {
            "lr": 10 ** trial.suggest_float("lr_log10", lr_lo, lr_hi),
            "betas": (trial.suggest_float("beta1", b1_lo, b1_hi), trial.suggest_float("beta2", b2_lo, b2_hi)),
            "weight_decay": 10 ** trial.suggest_float("wd_log10", wd_lo, wd_hi),
        }

    def _suggest_lamb_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Search space for LAMB."""
        lr_lo, lr_hi = self._range("lr_log10", -4, -1)
        b1_lo, b1_hi = self._range("beta1", 0.8, 0.99)
        b2_lo, b2_hi = self._range("beta2", 0.9, 0.999)
        wd_lo, wd_hi = self._range("wd_log10", -5, -2)
        return {
            "lr": 10 ** trial.suggest_float("lr_log10", lr_lo, lr_hi),
            "betas": (trial.suggest_float("beta1", b1_lo, b1_hi), trial.suggest_float("beta2", b2_lo, b2_hi)),
            "weight_decay": 10 ** trial.suggest_float("wd_log10", wd_lo, wd_hi),
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Runs one Optuna trial and returns the best validation accuracy."""
        set_random_seed(TUNER_SEED)
        model = self.model_class(**self.model_params).to(self.device)
        params = self._suggestion_methods[self.optimizer_name](trial)
        optimizer = self.optimizer_factory(model.parameters(), **params)
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_config=self.scheduler_config,
            total_epochs=self.epochs_per_trial,
            base_lr=params["lr"],
        )
        early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        for epoch in range(self.epochs_per_trial):
            train_one_epoch(model, optimizer, self.criterion, self.train_loader, self.device)
            val_metrics = evaluate_model(model, self.val_loader, self.device)
            if scheduler is not None:
                scheduler.step()
            if early_stopping.step(val_metrics, model, epoch):
                break

        return early_stopping.best_val_accuracy

    def get_best_params(self, study: optuna.Study) -> dict[str, Any]:
        """Reconstructs optimizer-ready params from the best trial."""
        return self._suggestion_methods[self.optimizer_name](study.best_trial)

    def tune(self, n_trials: int = 50, timeout: int | None = None) -> dict[str, Any]:
        """Runs hyperparameter search and returns the best optimizer params."""
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        best_params = self.get_best_params(study)
        logger.info(
            "Tuning completed for %s. Best accuracy: %.4f. Best params: %s",
            self.optimizer_name,
            study.best_value,
            best_params,
        )

        return best_params
