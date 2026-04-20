import torch.optim as optim

from src.data.noises import GaussianNoiseAdder, SaltAndPepperNoiseAdder
from src.experiment.robustness import run_comparative_robustness_evaluation
from src.experiment.runner import ExperimentRunner
from src.models.simple_cnn import SimpleCNN

MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
}

NOISE_REGISTRY = {
    "GaussianNoiseAdder": GaussianNoiseAdder,
    "SaltAndPepperNoiseAdder": SaltAndPepperNoiseAdder,
}

OPTIMIZER_REGISTRY = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

CONFIG_PATH = "configs/base_config.yaml"


def run_experiments() -> None:
    runner = ExperimentRunner(
        config_path=CONFIG_PATH,
        model_registry=MODEL_REGISTRY,
        noise_registry=NOISE_REGISTRY,
        optimizer_registry=OPTIMIZER_REGISTRY,
    )
    runner.run()


def run_robustness() -> None:
    run_comparative_robustness_evaluation(
        config_path=CONFIG_PATH,
        optimizer_registry=OPTIMIZER_REGISTRY,
    )


if __name__ == "__main__":
    run_experiments()
