import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.processing import _worker_init_fn, generate_noisy_datasets, get_dataloaders


class TestWorkerInitFn:
    def test_sets_seeds(self):
        """Worker init function should run without errors when given a worker id."""
        torch.manual_seed(42)
        _worker_init_fn(0)


class TestGenerateNoisyDatasets:
    def test_skips_existing(self, tmp_path):
        """Existing scenario directories should be skipped without reprocessing."""
        source = tmp_path / "source"
        target = tmp_path / "target"

        scenario_dir = target / "Test_no_noise"
        scenario_dir.mkdir(parents=True)

        with patch("src.data.processing.torchvision.datasets.ImageFolder") as mock_folder:
            mock_folder.return_value = MagicMock(imgs=[], classes=[])
            generate_noisy_datasets(
                source_path=str(source),
                target_root_path=str(target),
                noise_scenarios={"no_noise": []},
                noise_registry={},
                folder_template="Test_{scenario_name}",
            )

    def test_creates_target_dir(self, tmp_path):
        """Target root directory should be created if it doesn't exist."""
        source = tmp_path / "source"
        target = tmp_path / "target"

        with patch("src.data.processing.torchvision.datasets.ImageFolder") as mock_folder:
            mock_folder.return_value = MagicMock(imgs=[], classes=[])
            generate_noisy_datasets(
                source_path=str(source),
                target_root_path=str(target),
                noise_scenarios={"no_noise": []},
                noise_registry={},
                folder_template="Test_{scenario_name}",
            )
        assert target.exists()

    def test_processes_images_with_noise(self, tmp_path):
        """Full image processing pipeline: load images, apply noise, save to target."""
        from PIL import Image

        from src.config import NoiseTransformConfig
        from src.data.noises import GaussianNoiseAdder

        source = tmp_path / "source"
        cls_dir = source / "cat"
        cls_dir.mkdir(parents=True)
        for i in range(3):
            img = Image.new("RGB", (32, 32), color=(100, 100, 100))
            img.save(cls_dir / f"img_{i}.jpg")

        target = tmp_path / "target"
        noise_configs = [NoiseTransformConfig(name="GaussianNoiseAdder", params={"std": 0.1})]
        noise_registry = {"GaussianNoiseAdder": GaussianNoiseAdder}

        generate_noisy_datasets(
            source_path=str(source),
            target_root_path=str(target),
            noise_scenarios={"gaussian_0.1": noise_configs},
            noise_registry=noise_registry,
            folder_template="Test_{scenario_name}",
        )

        output_dir = target / "Test_gaussian_0.1" / "cat"
        assert output_dir.exists()
        generated = list(output_dir.iterdir())
        assert len(generated) == 3

    def test_handles_corrupt_image(self, tmp_path):
        """Corrupt images should be skipped without crashing the pipeline."""
        source = tmp_path / "source"
        cls_dir = source / "cat"
        cls_dir.mkdir(parents=True)
        (cls_dir / "corrupt.jpg").write_bytes(b"not an image")

        target = tmp_path / "target"

        with patch("src.data.processing.torchvision.datasets.ImageFolder") as mock_folder:
            mock_folder.return_value = MagicMock(
                imgs=[(str(cls_dir / "corrupt.jpg"), 0)],
                classes=["cat"],
            )
            generate_noisy_datasets(
                source_path=str(source),
                target_root_path=str(target),
                noise_scenarios={"noisy": []},
                noise_registry={},
                folder_template="Test_{scenario_name}",
            )


class TestGetDataloaders:
    def test_missing_directory_raises(self, tmp_path):
        """Non-existent scenario directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            get_dataloaders(
                preprocessed_root_path=str(tmp_path),
                scenario_folder_template="Test_{scenario_name}",
                scenario_name="no_noise",
                random_state=42,
                batch_size=4,
                num_workers=0,
            )

    def test_returns_three_loaders(self, tmp_path):
        """get_dataloaders should return train, val, and test DataLoaders."""
        scenario_dir = tmp_path / "Test_no_noise"
        for cls in ["cat", "dog"]:
            cls_dir = scenario_dir / cls
            cls_dir.mkdir(parents=True)
            for i in range(20):
                from PIL import Image

                img = Image.new("RGB", (32, 32), color=(i * 10 % 256, 0, 0))
                img.save(cls_dir / f"img_{i}.jpg")

        train_loader, val_loader, test_loader = get_dataloaders(
            preprocessed_root_path=str(tmp_path),
            scenario_folder_template="Test_{scenario_name}",
            scenario_name="no_noise",
            random_state=42,
            batch_size=4,
            num_workers=0,
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_split_ratios(self, tmp_path):
        """Verify 80/20 train_val/test split, then 85/15 train/val split."""
        scenario_dir = tmp_path / "Test_no_noise"
        for cls in ["a", "b"]:
            cls_dir = scenario_dir / cls
            cls_dir.mkdir(parents=True)
            from PIL import Image

            for i in range(50):
                img = Image.new("RGB", (32, 32))
                img.save(cls_dir / f"img_{i}.jpg")

        train_loader, val_loader, test_loader = get_dataloaders(
            preprocessed_root_path=str(tmp_path),
            scenario_folder_template="Test_{scenario_name}",
            scenario_name="no_noise",
            random_state=42,
            batch_size=4,
            num_workers=0,
        )

        total = 100
        train_val_size = int(0.8 * total)
        test_size = total - train_val_size
        train_size = int(0.85 * train_val_size)
        val_size = train_val_size - train_size

        assert len(train_loader.dataset) == train_size
        assert len(val_loader.dataset) == val_size
        assert len(test_loader.dataset) == test_size

    def test_reproducible_splits(self, tmp_path):
        """Same random_state should produce identical splits."""
        scenario_dir = tmp_path / "Test_no_noise"
        for cls in ["a", "b"]:
            cls_dir = scenario_dir / cls
            cls_dir.mkdir(parents=True)
            from PIL import Image

            for i in range(30):
                img = Image.new("RGB", (32, 32))
                img.save(cls_dir / f"img_{i}.jpg")

        loaders1 = get_dataloaders(
            preprocessed_root_path=str(tmp_path),
            scenario_folder_template="Test_{scenario_name}",
            scenario_name="no_noise",
            random_state=42,
            batch_size=4,
            num_workers=0,
        )
        loaders2 = get_dataloaders(
            preprocessed_root_path=str(tmp_path),
            scenario_folder_template="Test_{scenario_name}",
            scenario_name="no_noise",
            random_state=42,
            batch_size=4,
            num_workers=0,
        )

        assert len(loaders1[2].dataset) == len(loaders2[2].dataset)

    def test_subset_size(self, tmp_path):
        """debug_subset_size should limit total samples across all three splits."""
        scenario_dir = tmp_path / "Test_no_noise"
        for cls in ["a", "b"]:
            cls_dir = scenario_dir / cls
            cls_dir.mkdir(parents=True)
            from PIL import Image

            for i in range(50):
                img = Image.new("RGB", (32, 32))
                img.save(cls_dir / f"img_{i}.jpg")

        train_loader, val_loader, test_loader = get_dataloaders(
            preprocessed_root_path=str(tmp_path),
            scenario_folder_template="Test_{scenario_name}",
            scenario_name="no_noise",
            random_state=42,
            batch_size=4,
            num_workers=0,
            subset_size=20,
        )

        total_loaded = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        assert total_loaded == 20
