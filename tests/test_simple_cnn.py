import pytest
import torch

from src.models.simple_cnn import SimpleCNN


class TestSimpleCNN:
    def test_output_shape(self):
        """Forward pass should produce (batch_size, num_classes) output."""
        model = SimpleCNN(num_classes=10)
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_different_num_classes(self):
        """Output dimension should match the specified num_classes."""
        for nc in [2, 5, 100]:
            model = SimpleCNN(num_classes=nc)
            x = torch.rand(1, 3, 32, 32)
            out = model(x)
            assert out.shape == (1, nc)

    def test_different_input_sizes(self):
        """Adaptive pooling should handle various spatial dimensions."""
        model = SimpleCNN(num_classes=10)
        for size in [32, 64, 128]:
            x = torch.rand(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 10)

    def test_batch_sizes(self):
        """Batch dimension should pass through unchanged."""
        model = SimpleCNN(num_classes=10)
        for bs in [1, 4, 16]:
            x = torch.rand(bs, 3, 32, 32)
            out = model(x)
            assert out.shape[0] == bs

    def test_state_dict_keys(self):
        """Model should have both 'features' and 'classifier' parameter groups."""
        model = SimpleCNN(num_classes=10)
        keys = set(model.state_dict().keys())
        assert any("features" in k for k in keys)
        assert any("classifier" in k for k in keys)

    def test_gradients_flow(self):
        """All trainable parameters should receive gradients after backward pass."""
        model = SimpleCNN(num_classes=10)
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_eval_mode(self):
        """Model should work in eval mode with no_grad context."""
        model = SimpleCNN(num_classes=10)
        model.eval()
        with torch.no_grad():
            x = torch.rand(1, 3, 32, 32)
            out = model(x)
        assert out.shape == (1, 10)

    def test_no_nan_in_output(self):
        """Forward pass should not produce NaN or Inf values."""
        model = SimpleCNN(num_classes=10)
        x = torch.rand(4, 3, 32, 32)
        out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different logits."""
        model = SimpleCNN(num_classes=10)
        model.eval()
        torch.manual_seed(0)
        x1 = torch.rand(1, 3, 32, 32)
        torch.manual_seed(1)
        x2 = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert not torch.allclose(out1, out2), "Model ignores input — outputs are identical"

    def test_parameter_count_reasonable(self):
        """SimpleCNN should have a reasonable number of parameters (< 10M)."""
        model = SimpleCNN(num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 10_000_000, f"Too many params: {total_params}"
