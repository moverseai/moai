import torch

__all__ = ["Int32, Float32"]


class Int32(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.int()


class Float32(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float()
