import torch

__all__ = ["Clamp"]


class Clamp(torch.nn.Module):
    min_value: float
    max_value: float

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super(Clamp, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_value, max=self.max_value)
