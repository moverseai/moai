import torch

__all__ = ["Binary"]


class Binary(torch.nn.Module):
    def __init__(self):
        super(Binary, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor != 0.0).bool()  # TODO: byte or bool?
