import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch_optimizer import Lamb

from src.training.early_stopping import EarlyStopping
from src.training.evaluate import evaluate_model
from src.training.train import train_one_epoch
from src.types import ModelFactory


class HyperparameterTuner:
    def __init__(
        self,
        model_class: ModelFactory,
        model_params: dict,
        optimizer_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
        epochs_per_trial: int = 12,
        early_stopping_patience: int = 3,
    ):
        """
        Initializes the tuner.

        Args:
            model_class: Model class.
            optimizer_name: Name of the optimizer ('SGD', 'Adam', 'LAMB').
            train_loader: DataLoader for train data.
            val_loader: DataLoader for validation data.
            criterion: Loss function. Defaults to CrossEntropyLoss.
            epochs_per_trial: Max epochs per trial.
            early_stopping_patience: Stops a trial early if val accuracy doesn't improve.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epochs_per_trial = epochs_per_trial
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._suggestion_methods = {
            "SGD": self._suggest_sgd_params,
            "Adam": self._suggest_adam_params,
            "LAMB": self._suggest_lamb_params,
        }

        if optimizer_name not in self._suggestion_methods:
            raise ValueError(
                f"Optimizer '{optimizer_name}' doesn't support. " f"Available: {list(self._suggestion_methods.keys())}"
            )

        self.optimizer_name = optimizer_name
        print(f"The tuner will use the device: {self.device}")
        print(f"The tuner is configured to select parameters for {self.optimizer_name}")

    def _suggest_sgd_params(self, trial: optuna.Trial) -> dict:
        """Search space for SGD."""
        return {
            "lr": 10 ** trial.suggest_float("lr_log10", -4, 0),
            "momentum": trial.suggest_float("momentum", 0.5, 0.99),
            "nesterov": trial.suggest_categorical("nesterov", [False, True]),
            "weight_decay": 10 ** trial.suggest_float("wd_log10", -6, -3.3),
            "warmup_frac": trial.suggest_float("warmup_frac", 0.0, 0.1),
        }

    def _suggest_adam_params(self, trial: optuna.Trial) -> dict:
        """Search space for Adam."""
        return {
            "lr": 10 ** trial.suggest_float("lr_log10", -5, -2),
            "betas": (trial.suggest_float("adam_beta1", 0.8, 0.99), trial.suggest_float("adam_beta2", 0.9, 0.999)),
            "weight_decay": 10 ** trial.suggest_float("wd_log10", -6, -3),
        }

    def _suggest_lamb_params(self, trial: optuna.Trial) -> dict:
        """Search space for LAMB."""
        return {
            "lr": 10 ** trial.suggest_float("lr_log10", -4, -1),
            "betas": (trial.suggest_float("lamb_beta1", 0.8, 0.99), trial.suggest_float("lamb_beta2", 0.9, 0.999)),
            "weight_decay": 10 ** trial.suggest_float("wd_log10", -5, -2),
        }

    def _build_optimizer(self, model: nn.Module, params: dict) -> tuple[torch.optim.Optimizer, OneCycleLR | None]:
        """Creates optimizer and optional scheduler from suggested params."""
        optimizer: torch.optim.Optimizer
        scheduler = None
        if self.optimizer_name == "SGD":
            warmup_frac = params.pop("warmup_frac", 0.0)
            optimizer = optim.SGD(model.parameters(), **params)
            if warmup_frac > 0:
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=params["lr"],
                    epochs=self.epochs_per_trial,
                    steps_per_epoch=len(self.train_loader),
                    pct_start=warmup_frac,
                )
        elif self.optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), **params)
        else:
            optimizer = Lamb(model.parameters(), **params)
        return optimizer, scheduler

    def objective(self, trial: optuna.Trial) -> float:
        """Runs one Optuna trial and returns the best validation accuracy."""
        model = self.model_class(**self.model_params).to(self.device)
        params = self._suggestion_methods[self.optimizer_name](trial)
        optimizer, scheduler = self._build_optimizer(model, params)
        early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        for epoch in range(self.epochs_per_trial):
            train_one_epoch(model, optimizer, self.criterion, self.train_loader, self.device, scheduler)
            val_metrics = evaluate_model(model, self.val_loader, self.device)
            if early_stopping.step(val_metrics, model, epoch):
                break

        return early_stopping.best_val_accuracy

    def tune(self, n_trials: int = 100, direction: str = "maximize", timeout: int = None) -> optuna.Study:
        """
        Starts the hyperparameter selection process.

        Args:
            n_trials: Number of selection iterations.
            direction: Optimization direction ("maximize" or "minimize").
            timeout: Maximum time for selection in seconds.
        """
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        print("\nThe selection process is completed.")
        print(f"Best Accuracy for {self.optimizer_name}: {study.best_value:.4f}")
        print("With hyperparams:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")

        return study
