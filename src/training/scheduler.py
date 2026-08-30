import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from src.config import SchedulerConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: SchedulerConfig,
    total_epochs: int,
    base_lr: float,
) -> LRScheduler | None:
    """Creates an LR scheduler from config. Returns None for constant schedule without warmup."""
    warmup_epochs = (
        max(1, int(total_epochs * scheduler_config.warmup_ratio)) if scheduler_config.warmup_ratio > 0 else 0
    )
    main_epochs = total_epochs - warmup_epochs
    min_lr = base_lr * scheduler_config.min_lr_ratio

    main: LRScheduler | None = None

    if scheduler_config.name == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=min_lr)
    elif scheduler_config.name == "linear":
        main = LinearLR(optimizer, start_factor=1.0, end_factor=scheduler_config.min_lr_ratio, total_iters=main_epochs)
    elif scheduler_config.name == "constant":
        if warmup_epochs == 0:
            return None
    else:
        raise ValueError(f"Unknown scheduler: '{scheduler_config.name}'. Available: constant, cosine, linear")

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        if main is None:
            return warmup
        return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])

    return main
