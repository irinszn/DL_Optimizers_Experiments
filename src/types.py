from typing import Callable, TypeAlias

import torch
import torch.nn as nn
from torch.optim import Optimizer

OptimizerFactory: TypeAlias = Callable[..., Optimizer]
OptimizerRegistry: TypeAlias = dict[str, OptimizerFactory]

ModelFactory: TypeAlias = Callable[..., nn.Module]
ModelRegistry: TypeAlias = dict[str, ModelFactory]

NoiseTransform: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
NoiseFactory: TypeAlias = Callable[..., NoiseTransform]
NoiseRegistry: TypeAlias = dict[str, NoiseFactory]
