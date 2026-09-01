import logging
import random

import numpy as np
import torch

SPLIT_RANDOM_STATE = 42


def setup_logging(level: int = logging.INFO, log_file: str = "experiment.log") -> None:
    """
    Configures root logger to write to both console and a log file.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Path to the log file.
    """
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def parse_scenario(name: str) -> tuple[str, float]:
    """Parses a scenario name into (noise_type, level). E.g. 'gaussian_0.05' → ('gaussian', 0.05)."""
    if name == "no_noise":
        return "none", 0.0
    parts = name.rsplit("_", 1)
    return parts[0], float(parts[1])


def find_nearest_tuned_scenario(target: str, tuned_scenarios: list[str]) -> str:
    """Finds the tuned scenario closest to target by noise type and level."""
    target_type, target_level = parse_scenario(target)

    same_type = [(s, parse_scenario(s)[1]) for s in tuned_scenarios if parse_scenario(s)[0] == target_type]

    if same_type:
        return min(same_type, key=lambda x: abs(x[1] - target_level))[0]

    raise ValueError(
        f"No tuned scenario with noise type '{target_type}' found for scenario '{target}'. "
        f"Tuned scenarios: {tuned_scenarios}. "
        f"Add a scenario of the same noise type to tune_scenarios in config."
    )


def set_random_seed(seed: int) -> None:
    """
    Sets the seed for all major random number generators to ensure reproducibility.

    Args:
        seed: The integer value to use as the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
