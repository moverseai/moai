import functools

import torch

__all__ = [
    "Downsample2d",
]


class Downsample2d(torch.nn.Module):
    def __init__(self, scale: float = 0.5, mode: str = "bilinear"):
        super(Downsample2d, self).__init__()
        self.downsample = functools.partial(
            torch.nn.functional.interpolate, scale_factor=scale, mode=mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)
