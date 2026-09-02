import torch
import pytest

from src.training.train import train_one_epoch


class TestTrainOneEpoch:
    def test_returns_float_loss(self, model, synthetic_dataloader, device):
        """Should return a positive float loss value."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        loss = train_one_epoch(model, optimizer, criterion, synthetic_dataloader, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_is_average(self, model, synthetic_dataloader, device):
        """Returned loss should be average per batch, not sum (CE with 10 classes ~ log(10) ~ 2.3)."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        loss = train_one_epoch(model, optimizer, criterion, synthetic_dataloader, device)
        assert loss < 5.0

    def test_model_weights_change(self, model, synthetic_dataloader, device):
        """At least some model parameters should change after one epoch."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        initial_params = {k: v.clone() for k, v in model.named_parameters()}
        train_one_epoch(model, optimizer, criterion, synthetic_dataloader, device)

        changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                changed = True
                break
        assert changed

    def test_model_in_train_mode(self, model, synthetic_dataloader, device):
        """Model should be in training mode after train_one_epoch, even if started in eval."""
        model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        train_one_epoch(model, optimizer, criterion, synthetic_dataloader, device)
        assert model.training

    def test_multiple_epochs_loss_changes(self, model, synthetic_dataloader, device):
        """Loss should not be identical across consecutive epochs."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        for _ in range(3):
            loss = train_one_epoch(model, optimizer, criterion, synthetic_dataloader, device)
            losses.append(loss)
        assert not all(losses[i] == losses[i + 1] for i in range(len(losses) - 1))

    def test_loss_decreases_over_epochs(self, device):
        """Training should reduce loss over multiple epochs."""
        torch.manual_seed(0)
        from src.models.simple_cnn import SimpleCNN

        model = SimpleCNN(num_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        images = torch.rand(32, 3, 32, 32)
        labels = torch.randint(0, 10, (32,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        losses = []
        for _ in range(10):
            loss = train_one_epoch(model, optimizer, criterion, loader, device)
            losses.append(loss)

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
