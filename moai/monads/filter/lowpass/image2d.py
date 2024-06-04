import logging
import typing

import torch
from kornia.filters import GaussianBlur2d

log = logging.getLogger(__name__)

__all__ = ["Gaussian2d"]

# NOTE: https://dsp.stackexchange.com/questions/10057/gaussian-blur-standard-deviation-radius-and-kernel-size


class Gaussian2d(GaussianBlur2d):
    def __init__(
        self,
        kernel_size: typing.Tuple[int, int] = (5, 5),
        sigma: typing.Tuple[float, float] = (1.0, 1.0),
        border_type: str = "reflect",
    ):
        super(Gaussian2d, self).__init__(
            kernel_size=tuple(kernel_size), sigma=tuple(sigma), border_type=border_type
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return super(Gaussian2d, self).forward(image)
