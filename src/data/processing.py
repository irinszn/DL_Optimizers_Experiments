import logging
import os
import random
import shutil
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from src.config import load_config
from src.types import NoiseRegistry

logger = logging.getLogger(__name__)


def generate_noisy_datasets(
    source_path: str,
    target_root_path: str,
    noise_scenarios: dict,
    noise_registry: NoiseRegistry,
    folder_template: str,
) -> None:
    """
    Generates noisy datasets from a clean image folder and saves them to disk.
    Skips scenarios that already exist. Platform-agnostic: local, server, or Colab.

    Args:
        source_path: Path to the clean dataset (ImageFolder format).
        target_root_path: Root directory where noisy datasets will be saved.
        noise_scenarios: Scenario configs, e.g. {'gaussian_0.05': [{'name': 'GaussianNoiseAdder', 'params': {...}}]}.
        noise_registry: Dict mapping noise class names to their classes.
        folder_template: Format string for subfolder names, e.g. 'Animals10_{scenario_name}'.
    """
    logger.info("Loading clean dataset from: %s", source_path)
    clean_data = torchvision.datasets.ImageFolder(root=source_path)

    os.makedirs(target_root_path, exist_ok=True)
    logger.info("Generating datasets in: %s", target_root_path)

    for scenario_name, noise_config in noise_scenarios.items():
        scenario_folder_name = folder_template.format(scenario_name=scenario_name)
        target_path = os.path.join(target_root_path, scenario_folder_name)

        if os.path.exists(target_path):
            logger.info("Dataset for '%s' already exists. Skipping.", scenario_name)
            continue

        logger.info("Generating dataset '%s' in '%s'...", scenario_name, target_path)

        noise_transforms = [noise_registry[n.name](**n.params) for n in noise_config]
        transform_pipeline = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor(), *noise_transforms, transforms.ToPILImage()]
        )

        os.makedirs(target_path, exist_ok=True)
        for img_path, label_idx in tqdm(clean_data.imgs, desc=f"  Scenario {scenario_name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                processed_img = transform_pipeline(img)

                class_name = clean_data.classes[label_idx]
                class_path = os.path.join(target_path, class_name)
                os.makedirs(class_path, exist_ok=True)

                img_name = os.path.basename(img_path)
                processed_img.save(os.path.join(class_path, img_name))

            except Exception as e:
                logger.warning("Failed to process file %s: %s", img_path, e)

        logger.info("Done: %s", target_path)

    logger.info("Dataset generation completed.")


def generate_datasets_on_drive(config_path: str, noise_registry: NoiseRegistry) -> None:
    """
    Colab + Google Drive wrapper around generate_noisy_datasets.
    Generates each scenario locally in /content/tmp_* first (fast SSD),
    then copies to Drive and waits for Drive to sync before proceeding.

    Args:
        config_path: Path to the YAML configuration file.
        noise_registry: Dict mapping noise class names to their classes.
    """
    config = load_config(config_path)

    source_path = config.data.clean_data_path
    root_path = config.data.preprocessed_root_path
    folder_template = config.data.scenario_folder_template
    noise_scenarios = config.grid_search.noise_scenarios

    logger.info("Loading clean dataset from: %s", source_path)
    clean_data = torchvision.datasets.ImageFolder(root=source_path)

    os.makedirs(root_path, exist_ok=True)
    logger.info("Check and generate datasets in Drive path: %s", root_path)

    for scenario_name, noise_config in noise_scenarios.items():
        scenario_folder_name = folder_template.format(scenario_name=scenario_name)
        target_path = os.path.join(root_path, scenario_folder_name)

        if os.path.exists(target_path):
            logger.info("Dataset for '%s' already exists on Drive. Skipping.", scenario_name)
            continue

        local_temp = f"/content/tmp_{scenario_folder_name}"
        if os.path.exists(local_temp):
            shutil.rmtree(local_temp)
        os.makedirs(local_temp, exist_ok=True)

        logger.info("Generating dataset '%s' locally in '%s'...", scenario_name, local_temp)

        noise_transforms = [noise_registry[n.name](**n.params) for n in noise_config]
        transform_pipeline = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor(), *noise_transforms, transforms.ToPILImage()]
        )

        for img_path, label_idx in tqdm(clean_data.imgs, desc=f"  Scenario {scenario_name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                processed_img = transform_pipeline(img)

                class_name = clean_data.classes[label_idx]
                local_class_path = os.path.join(local_temp, class_name)
                os.makedirs(local_class_path, exist_ok=True)

                img_name = os.path.basename(img_path)
                processed_img.save(os.path.join(local_class_path, img_name))

            except Exception as e:
                logger.warning("Failed to process file %s: %s", img_path, e)

        logger.info("Uploading scenario '%s' to Google Drive...", scenario_name)
        shutil.copytree(local_temp, target_path)
        logger.info("Uploaded: %s", target_path)

        sync_seconds = 900
        logger.info("Waiting %d minutes for Google Drive to sync...", sync_seconds // 60)
        for _ in tqdm(range(sync_seconds), desc="Google Drive syncing"):
            time.sleep(1)

        logger.info("Synchronization completed. Proceeding to next scenario.")
        shutil.rmtree(local_temp)

    logger.info("Dataset verification and generation completed.")


def _worker_init_fn(worker_id: int) -> None:
    """Initializes each DataLoader worker with a unique deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    preprocessed_root_path: str,
    scenario_folder_template: str,
    scenario_name: str,
    random_state: int,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = False,
    subset_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads a preprocessed dataset from disk, splits into train/val/test, returns DataLoaders.
    The split is reproducible via random_state — all optimizers see identical data partitions.

    Args:
        preprocessed_root_path: Root directory with all preprocessed datasets.
        scenario_folder_template: Format string for the subfolder name, e.g. 'Animals10_{scenario_name}'.
        scenario_name: Scenario to load, e.g. 'gaussian_0.03'.
        random_state: Seed for reproducible train/val/test splits.
        batch_size: Batch size for all loaders.
        num_workers: Number of worker processes for data loading.
        pin_memory: If true, tensors are pinned to GPU memory for faster transfer.
        subset_size: If set, use a random subset of this size instead of the full dataset.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        FileNotFoundError: If the scenario directory does not exist.
    """
    scenario_folder_name = scenario_folder_template.format(scenario_name=scenario_name)
    data_path = os.path.join(preprocessed_root_path, scenario_folder_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Directory for scenario '{scenario_name}' not found at the expected path: {data_path}\n"
            f"Please ensure you have run the preprocessing script to generate this dataset."
        )

    logger.info("Loading data from: %s", data_path)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    if subset_size:
        logger.info("DEBUG MODE: Running on a slice of %d / %d samples", subset_size, len(full_dataset))
        subset_size = min(subset_size, len(full_dataset))
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(full_dataset), size=subset_size, replace=False).tolist()
        full_dataset = Subset(full_dataset, indices)

    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_data, test_data = random_split(
        full_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(random_state)
    )

    train_size = int(0.85 * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = random_split(
        train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(random_state)
    )

    generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader, test_loader
