import torch


class GaussianNoiseAdder:
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor + torch.randn_like(tensor) * self.std + self.mean
        return x.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class SaltAndPepperNoiseAdder:
    def __init__(self, amount: float = 0.04):
        self.amount = amount

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noisy = tensor.clone()
        c, h, w = noisy.size()
        num = int(self.amount * h * w)

        if num == 0:
            return noisy

        idx = torch.randperm(h * w)[:num]
        r, col = idx // w, idx % w
        half = num // 2
        noisy[:, r[:half], col[:half]] = 1.0
        noisy[:, r[half:], col[half:]] = 0.0
        return noisy.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(amount={self.amount})"
