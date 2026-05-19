from collections.abc import Callable, Mapping
from typing import TypeAlias

import torch
import torch.nn as nn
from torch.optim import Optimizer

OptimizerFactory: TypeAlias = Callable[..., Optimizer]
OptimizerRegistry: TypeAlias = Mapping[str, OptimizerFactory]

ModelFactory: TypeAlias = Callable[..., nn.Module]
ModelRegistry: TypeAlias = Mapping[str, ModelFactory]

NoiseTransform: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
NoiseFactory: TypeAlias = Callable[..., NoiseTransform]
NoiseRegistry: TypeAlias = Mapping[str, NoiseFactory]
