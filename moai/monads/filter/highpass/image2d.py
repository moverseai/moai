import functools
import logging

import torch
from kornia.filters import Laplacian, SpatialGradient

from moai.utils.arguments import assert_choices, assert_numeric

log = logging.getLogger(__name__)

__all__ = ["Laplacian2d", "Sobel2d", "Diff2d"]


class Image2d(SpatialGradient):

    __MODES__ = ["sobel", "diff"]
    __REDUCTIONS__ = ["none", "magnitude", "absum"]

    def __init__(
        self,
        mode: str = "sobel",  # one of ['sobel', 'diff']
        order: int = 1,
        normalized: bool = True,
        reduction: str = "magnitude",
    ):
        assert_numeric(log, "order", order, min_value=1, max_value=2)
        assert_choices(log, "mode", mode, Image2d.__MODES__)
        assert_choices(log, "reduction", reduction, Image2d.__REDUCTIONS__)
        super(Image2d, self).__init__(mode=mode, order=order, normalized=normalized)

        def abs_sum(t: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.sum(t.abs(), dim=[2])

        self.reduction_func = (
            functools.partial(abs_sum, dim=2)
            if reduction == "abssum"
            else (
                functools.partial(torch.linalg.norm, ord=2, dim=2)
                if reduction == "magnitude"
                else lambda t: t
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        filtered = super(Image2d, self).forward(image)
        return self.reduction_func(filtered)


Sobel2d = functools.partial(Image2d, mode="sobel")
Diff2d = functools.partial(Image2d, mode="diff")


class Laplacian2d(Laplacian):
    def __init__(
        self,
        kernel_size: int = 5,
        border_type: str = "reflect",
        normalized: bool = True,
    ):
        super(Laplacian2d, self).__init__(
            kernel_size=kernel_size, border_type=border_type, normalized=normalized
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return super(Laplacian2d, self).forward(image)
