import random

import numpy as np
import pytest
import torch

from src.utils import find_nearest_tuned_scenario, parse_scenario, set_random_seed, setup_logging


class TestParseScenario:
    def test_no_noise(self):
        """'no_noise' should parse as noise type 'none' with level 0.0."""
        assert parse_scenario("no_noise") == ("none", 0.0)

    def test_gaussian(self):
        """Gaussian scenario string should parse into type and float level."""
        assert parse_scenario("gaussian_0.05") == ("gaussian", 0.05)

    def test_salt_and_pepper(self):
        """Multi-word noise type with underscores should parse correctly."""
        noise_type, level = parse_scenario("salt_and_pepper_0.1")
        assert noise_type == "salt_and_pepper"
        assert level == pytest.approx(0.1)

    def test_zero_level(self):
        """Zero noise level should be valid and parse as 0.0."""
        assert parse_scenario("gaussian_0.0") == ("gaussian", 0.0)

    def test_high_level(self):
        """High noise levels close to 1.0 should parse correctly."""
        assert parse_scenario("gaussian_0.9") == ("gaussian", 0.9)

    def test_integer_level(self):
        """Integer noise level without decimal point should parse as float."""
        assert parse_scenario("gaussian_1") == ("gaussian", 1.0)


class TestFindNearestTunedScenario:
    def test_exact_match(self):
        """Exact scenario match should be returned directly."""
        tuned = ["gaussian_0.05", "gaussian_0.1", "no_noise"]
        assert find_nearest_tuned_scenario("gaussian_0.05", tuned) == "gaussian_0.05"

    def test_nearest_by_level(self):
        """Should return the tuned scenario with the closest noise level."""
        tuned = ["gaussian_0.01", "gaussian_0.1"]
        result = find_nearest_tuned_scenario("gaussian_0.08", tuned)
        assert result == "gaussian_0.1"

    def test_no_noise_match(self):
        """'no_noise' query should match 'no_noise' in the tuned list."""
        tuned = ["no_noise", "gaussian_0.1"]
        assert find_nearest_tuned_scenario("no_noise", tuned) == "no_noise"

    def test_no_matching_type_raises(self):
        """Missing noise type in tuned list should raise ValueError."""
        tuned = ["gaussian_0.05"]
        with pytest.raises(ValueError, match="No tuned scenario with noise type"):
            find_nearest_tuned_scenario("salt_and_pepper_0.1", tuned)

    def test_empty_tuned_raises(self):
        """Empty tuned scenario list should raise ValueError."""
        with pytest.raises(ValueError):
            find_nearest_tuned_scenario("gaussian_0.05", [])

    def test_same_type_different_levels(self):
        """Should pick the closest level among scenarios of the same noise type."""
        tuned = ["salt_and_pepper_0.02", "salt_and_pepper_0.1"]
        result = find_nearest_tuned_scenario("salt_and_pepper_0.05", tuned)
        assert result == "salt_and_pepper_0.02"

    def test_equidistant_scenarios(self):
        """When two scenarios are equidistant, result is deterministic (depends on float precision)."""
        tuned = ["gaussian_0.04", "gaussian_0.06"]
        result = find_nearest_tuned_scenario("gaussian_0.05", tuned)
        assert result in ("gaussian_0.04", "gaussian_0.06")


class TestSetRandomSeed:
    def test_reproducibility(self):
        """Same seed should produce identical random values across all RNG backends."""
        set_random_seed(42)
        a1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        set_random_seed(42)
        a2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert a1 == a2
        assert n1 == n2
        assert t1 == t2

    def test_different_seeds_differ(self):
        """Different seeds should produce different random tensors."""
        set_random_seed(1)
        v1 = torch.rand(5)

        set_random_seed(2)
        v2 = torch.rand(5)

        assert not torch.allclose(v1, v2)


class TestSetupLogging:
    def test_creates_log_file(self, tmp_path):
        """setup_logging should create a log file at the specified path."""
        import logging

        log_file = str(tmp_path / "test.log")
        root = logging.getLogger()
        original_handlers = root.handlers[:]

        setup_logging(log_file=log_file)

        logger = logging.getLogger("test_setup")
        logger.info("test message")

        import os

        assert os.path.exists(log_file)

        root.handlers = original_handlers

    def test_custom_level(self, tmp_path):
        """setup_logging with custom level should set root logger to that level."""
        import logging

        log_file = str(tmp_path / "test_level.log")
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level

        root.handlers = []
        root.setLevel(logging.WARNING)

        setup_logging(level=logging.DEBUG, log_file=log_file)
        assert root.level == logging.DEBUG

        root.handlers = original_handlers
        root.setLevel(original_level)
