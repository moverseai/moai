import torch
import functools
import logging

log = logging.getLogger(__name__)

__all__ = [
    "Adaptive",
]

class Adaptive(torch.nn.Module):
    def __init__(self,
        scale_factor:       float=2.0,  # downscale factor, divides the resolution
        mode:               str='max',  # 'max' or 'avg' pool
        dims:               int=2,      # 1, 2 or 3 dimensional downsampling
    ):
        super(Adaptive, self).__init__()
        self.pool_func = getattr(torch.nn.functional, f"adaptive_{mode}_pool{dims}d")
        self.scale_factor = scale_factor
        self.dims = dims

    def even_size(self, size: int, scale_factor: float) -> int:
        downscaled = int(size // scale_factor)
        return downscaled + int(downscaled % 2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        size = [self.even_size(s, self.scale_factor) for s in tensor.shape[2:2+ self.dims]]
        return self.pool_func(tensor, size)

Adaptive1d = functools.partial(Adaptive, dims=1)
Adaptive2d = functools.partial(Adaptive, dims=2)
Adaptive3d = functools.partial(Adaptive, dims=3)

AdaptiveMaxPool1d = functools.partial(Adaptive, dims=1, mode='max')
AdaptiveMaxPool2d = functools.partial(Adaptive, dims=2, mode='max')
AdaptiveMaxPool3d = functools.partial(Adaptive, dims=3, mode='max')

AdaptiveAvgPool1d = functools.partial(Adaptive, dims=1, mode='avg')
AdaptiveAvgPool2d = functools.partial(Adaptive, dims=2, mode='avg')
AdaptiveAvgPool3d = functools.partial(Adaptive, dims=3, mode='avg')

GlobalAvgPool1d = functools.partial(torch.nn.AdaptiveAvgPool1d, output_size=1)
GlobalAvgPool2d = functools.partial(torch.nn.AdaptiveAvgPool2d, output_size=1)
GlobalAvgPool3d = functools.partial(torch.nn.AdaptiveAvgPool3d, output_size=1)
