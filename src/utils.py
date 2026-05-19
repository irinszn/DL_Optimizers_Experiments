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
