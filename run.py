import argparse

import torch.optim as optim
from torch_optimizer import Lamb

from src.config import load_config
from src.data.noises import GaussianNoiseAdder, SaltAndPepperNoiseAdder
from src.data.processing import generate_noisy_datasets
from src.experiment.robustness import run_comparative_robustness_evaluation
from src.experiment.runner import ExperimentRunner
from src.models.simple_cnn import SimpleCNN
from src.utils import setup_logging

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
    "LAMB": Lamb,
}


def run_generate(config_path: str) -> None:
    config = load_config(config_path)
    generate_noisy_datasets(
        source_path=config.data.clean_data_path,
        target_root_path=config.data.preprocessed_root_path,
        noise_scenarios=config.grid_search.noise_scenarios,
        noise_registry=NOISE_REGISTRY,
        folder_template=config.data.scenario_folder_template,
    )


def run_experiments(config_path: str) -> None:
    runner = ExperimentRunner(
        config_path=config_path,
        model_registry=MODEL_REGISTRY,
        optimizer_registry=OPTIMIZER_REGISTRY,
    )
    runner.run()


def run_robustness(config_path: str) -> None:
    run_comparative_robustness_evaluation(
        config_path=config_path,
        optimizer_registry=OPTIMIZER_REGISTRY,
    )


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "experiments", "robustness", "generate"],
        help="all — generate + train + robustness, generate — create noisy datasets, experiments — only train, robustness — only evaluate",
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_generate(args.config)
        run_experiments(args.config)
        run_robustness(args.config)
    elif args.mode == "generate":
        run_generate(args.config)
    elif args.mode == "experiments":
        run_experiments(args.config)
    else:
        run_robustness(args.config)
