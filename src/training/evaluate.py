import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict[str, float]:
    all_labels: list[int] = []
    all_predictions: list[int] = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions) * 100,
        "precision": precision_score(all_labels, all_predictions, average="macro", zero_division=0) * 100,
        "recall": recall_score(all_labels, all_predictions, average="macro", zero_division=0) * 100,
        "f1_score": f1_score(all_labels, all_predictions, average="macro", zero_division=0) * 100,
    }
