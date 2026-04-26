import random

import numpy as np
import torch

SPLIT_RANDOM_STATE = 42


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
