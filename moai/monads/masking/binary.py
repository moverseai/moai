import torch

__all__ = ["Binary"]

class Binary(torch.nn.Module):
    def __init__(self):
        super(Binary, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x != 0.0).byte() #TODO: byte or bool?