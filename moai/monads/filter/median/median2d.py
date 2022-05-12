from kornia.filters import (
    MedianBlur
)

import torch
import logging
import functools
import typing

log = logging.getLogger(__name__)

__all__ = ["Median2d"]


class Median2d(MedianBlur):
    def __init__(self,
        kernel_size:                typing.Tuple[int, int]=(5, 5),
        # border_type:                str='reflect'
    ):
        super(Median2d, self).__init__(
            kernel_size=tuple(kernel_size),
            # border_type=border_type
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 4:
            image = image.unsqueeze(0).clone()
        return super(Median2d, self).forward(image)