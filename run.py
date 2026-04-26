import argparse

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


def run_experiments(config_path: str) -> None:
    runner = ExperimentRunner(
        config_path=config_path,
        model_registry=MODEL_REGISTRY,
        noise_registry=NOISE_REGISTRY,
        optimizer_registry=OPTIMIZER_REGISTRY,
    )
    runner.run()


def run_robustness(config_path: str) -> None:
    run_comparative_robustness_evaluation(
        config_path=config_path,
        optimizer_registry=OPTIMIZER_REGISTRY,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "experiments", "robustness"],
        help="all — run experiments then robustness, experiments — only train, robustness — only evaluate",
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_experiments(args.config)
        run_robustness(args.config)
    elif args.mode == "experiments":
        run_experiments(args.config)
    else:
        run_robustness(args.config)
