import torch
import pytest

from src.data.noises import GaussianNoiseAdder, SaltAndPepperNoiseAdder


class TestGaussianNoiseAdder:
    def test_output_range(self, sample_tensor):
        """Output values must be clamped to [0, 1]."""
        adder = GaussianNoiseAdder(mean=0.0, std=0.3)
        result = adder(sample_tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shape_preserved(self, sample_tensor):
        """Noise should not change tensor shape."""
        adder = GaussianNoiseAdder(std=0.1)
        result = adder(sample_tensor)
        assert result.shape == sample_tensor.shape

    def test_noise_applied(self, sample_tensor):
        """Non-zero std should modify the tensor."""
        torch.manual_seed(0)
        adder = GaussianNoiseAdder(std=0.5)
        result = adder(sample_tensor)
        assert not torch.allclose(result, sample_tensor)

    def test_zero_std_no_change(self, sample_tensor):
        """Zero std with zero mean should return the original tensor."""
        adder = GaussianNoiseAdder(mean=0.0, std=0.0)
        result = adder(sample_tensor)
        assert torch.allclose(result, sample_tensor)

    def test_repr(self):
        """String representation should contain class name and parameters."""
        adder = GaussianNoiseAdder(mean=0.1, std=0.2)
        assert "GaussianNoiseAdder" in repr(adder)
        assert "0.1" in repr(adder)
        assert "0.2" in repr(adder)

    def test_high_std_clamps(self):
        """Extreme std values should still produce output in [0, 1]."""
        tensor = torch.ones(3, 8, 8) * 0.5
        adder = GaussianNoiseAdder(mean=0.0, std=10.0)
        result = adder(tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestSaltAndPepperNoiseAdder:
    def test_output_range(self, sample_tensor):
        """Output values must stay in [0, 1]."""
        adder = SaltAndPepperNoiseAdder(amount=0.1)
        result = adder(sample_tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shape_preserved(self, sample_tensor):
        """Noise should not change tensor shape."""
        adder = SaltAndPepperNoiseAdder(amount=0.1)
        result = adder(sample_tensor)
        assert result.shape == sample_tensor.shape

    def test_zero_amount_no_change(self, sample_tensor):
        """Zero amount should return an identical tensor."""
        adder = SaltAndPepperNoiseAdder(amount=0.0)
        result = adder(sample_tensor)
        assert torch.equal(result, sample_tensor)

    def test_pixels_modified(self, sample_tensor):
        """Non-zero amount should modify some pixels."""
        torch.manual_seed(0)
        adder = SaltAndPepperNoiseAdder(amount=0.1)
        result = adder(sample_tensor)
        assert not torch.equal(result, sample_tensor)

    def test_salt_and_pepper_values(self):
        """Result should contain both 1.0 (salt) and 0.0 (pepper) pixels."""
        torch.manual_seed(0)
        tensor = torch.ones(3, 16, 16) * 0.5
        adder = SaltAndPepperNoiseAdder(amount=0.5)
        result = adder(tensor)
        has_ones = (result == 1.0).any()
        has_zeros = (result == 0.0).any()
        assert has_ones
        assert has_zeros

    def test_repr(self):
        """String representation should contain class name and amount."""
        adder = SaltAndPepperNoiseAdder(amount=0.04)
        assert "SaltAndPepperNoiseAdder" in repr(adder)
        assert "0.04" in repr(adder)

    def test_small_image(self):
        """Noise should work on very small images without errors."""
        tensor = torch.rand(3, 2, 2)
        adder = SaltAndPepperNoiseAdder(amount=0.5)
        result = adder(tensor)
        assert result.shape == tensor.shape

    def test_salt_and_pepper_pixel_counts(self):
        """Exact number of salt and pepper pixels should match the amount/split."""
        torch.manual_seed(42)
        h, w = 32, 32
        tensor = torch.ones(3, h, w) * 0.5
        amount = 0.1
        adder = SaltAndPepperNoiseAdder(amount=amount)
        result = adder(tensor)

        num = int(amount * h * w)
        half = num // 2
        salt_mask = (result[0] == 1.0)
        pepper_mask = (result[0] == 0.0)
        assert salt_mask.sum().item() == half
        assert pepper_mask.sum().item() == num - half

    def test_salt_and_pepper_determinism(self):
        """Same seed should produce identical noise."""
        tensor = torch.ones(3, 16, 16) * 0.5
        torch.manual_seed(7)
        r1 = SaltAndPepperNoiseAdder(amount=0.2)(tensor)
        torch.manual_seed(7)
        r2 = SaltAndPepperNoiseAdder(amount=0.2)(tensor)
        assert torch.equal(r1, r2)


class TestGaussianNoiseBehavior:
    def test_gaussian_noise_distribution(self):
        """Noise on unclamped pixels should have approximately correct mean and std."""
        torch.manual_seed(0)
        tensor = torch.ones(3, 64, 64) * 0.5
        std = 0.15
        adder = GaussianNoiseAdder(mean=0.0, std=std)
        result = adder(tensor)
        diff = result - tensor
        mask = (result > 0.0) & (result < 1.0)
        unclamped_diff = diff[mask]
        assert unclamped_diff.mean().item() == pytest.approx(0.0, abs=0.02)
        assert unclamped_diff.std().item() == pytest.approx(std, abs=0.03)

    def test_gaussian_determinism(self):
        """Same seed should produce identical noise."""
        tensor = torch.ones(3, 16, 16) * 0.5
        torch.manual_seed(5)
        r1 = GaussianNoiseAdder(std=0.2)(tensor)
        torch.manual_seed(5)
        r2 = GaussianNoiseAdder(std=0.2)(tensor)
        assert torch.equal(r1, r2)

    def test_gaussian_nonzero_mean(self):
        """With std=0, result should equal (tensor + mean).clamp(0, 1)."""
        tensor = torch.rand(3, 8, 8)
        adder = GaussianNoiseAdder(mean=0.3, std=0.0)
        result = adder(tensor)
        expected = (tensor + 0.3).clamp(0.0, 1.0)
        assert torch.allclose(result, expected)
