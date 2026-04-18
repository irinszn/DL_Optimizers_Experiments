import os
import shutil
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm


def generate_datasets_on_drive(config_path: str, noise_registry: dict) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    source_path = config["data"]["clean_data_path"]
    root_path = config["data"]["preprocessed_root_path"]
    folder_template = config["data"]["scenario_folder_template"]

    print(f"Loading clean dataset from: {source_path}")
    clean_data = torchvision.datasets.ImageFolder(root=source_path)

    os.makedirs(root_path, exist_ok=True)
    print(f"Check and generate datasets in Drive path: {root_path}")

    for scenario_name, noise_config in config["grid_search"]["noise_scenarios"].items():
        scenario_folder_name = folder_template.format(scenario_name=scenario_name)
        target_path = os.path.join(root_path, scenario_folder_name)

        if os.path.exists(target_path):
            print(f"Dataset for '{scenario_name}' already exists on Drive. Skip.")
            continue

        local_temp = f"/content/tmp_{scenario_folder_name}"
        if os.path.exists(local_temp):
            shutil.rmtree(local_temp)
        os.makedirs(local_temp, exist_ok=True)

        print(f"\nGenerating dataset '{scenario_name}' locally in '{local_temp}'...")

        noise_transforms = [noise_registry[n["name"]](**n["params"]) for n in noise_config]

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
                print(f"Failed to process file {img_path}: {e}")

        print(f"Uploading scenario '{scenario_name}' to Google Drive...")
        shutil.copytree(local_temp, target_path)
        print(f"Uploaded: {target_path}")

        sync_seconds = 900
        print(f"Waiting {sync_seconds // 60} minutes for Google Drive to sync...")
        for _ in tqdm(range(sync_seconds), desc="Google Drive syncing"):
            time.sleep(1)

        print("Synchronization time completed. Proceeding to next scenario.")
        shutil.rmtree(local_temp)

    print("\nDataset verification and generation are completed.")


def get_dataloaders_from_drive(
    preprocessed_root_path: str,
    scenario_folder_template: str,
    scenario_name: str,
    random_state: int,
    batch_size: int,
    subset_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    scenario_folder_name = scenario_folder_template.format(scenario_name=scenario_name)
    data_path = os.path.join(preprocessed_root_path, scenario_folder_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Directory for scenario '{scenario_name}' not found at the expected path: {data_path}\n"
            f"Please ensure you have run the preprocessing script to generate this dataset."
        )

    print(f"    - Loading data from: {data_path}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    if subset_size:
        print(f"    - DEBUG MODE: Running on a slice of {subset_size} / {len(full_dataset)} samples\n")
        subset_size = min(subset_size, len(full_dataset))
        indices = list(range(subset_size))
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

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
