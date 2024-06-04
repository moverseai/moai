import functools
import typing

import torch

__all__ = [
    "Upsample2d",
]


class Upsample2d(torch.nn.Module):
    def __init__(
        self,
        resolution: typing.Sequence[int] = None,
        scale: float = 2.0,
        mode: str = "bilinear",
    ):
        super(Upsample2d, self).__init__()
        if resolution:
            self.upsample = functools.partial(
                torch.nn.functional.interpolate, size=tuple(resolution), mode=mode
            )
        else:
            self.upsample = functools.partial(
                torch.nn.functional.interpolate, scale_factor=scale, mode=mode
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)
