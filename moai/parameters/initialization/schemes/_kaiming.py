import functools
import logging
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)

try:
    from mapped_convolution.nn import MappedConvolution, MappedTransposedConvolution
except ImportError:
    __SPHERICAL_MAPPED_INCLUDED__ = False
else:
    __SPHERICAL_MAPPED_INCLUDED__ = True

__all__ = ["Kaiming"]


class Kaiming(
    Callable
):  # TODO: should break because it needs to be fixed to call module.apply()
    def __init__(
        self,
        zero_bias: bool = False,
        gain: float = 0.0,
        mode: str = "fan_out",  # one of ['fan_in', 'fan_out']
        nonlinearity: str = "relu",  # one of ['relu', 'leaky_relu']
    ):
        self.conv_w_init = functools.partial(
            torch.nn.init.kaiming_normal_, a=gain, mode=mode, nonlinearity=nonlinearity
        )
        self.conv_b_init = torch.nn.init.zeros_ if zero_bias else torch.nn.init.normal_
        self.w_init = torch.nn.init.normal_

    def __call__(self, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Conv1d):
            self.w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            self.conv_w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.Conv3d):
            self.conv_w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            self.w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            self.w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose3d):
            self.w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.BatchNorm1d):
            torch.nn.init.normal_(module.weight, mean=1, std=0.02)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight, mean=1, std=0.02)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.BatchNorm3d):
            torch.nn.init.normal_(module.weight, mean=1, std=0.02)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.Linear):
            self.conv_w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif __SPHERICAL_MAPPED_INCLUDED__ and (
            isinstance(module, MappedConvolution)
            or isinstance(module, MappedTransposedConvolution)
        ):
            self.conv_w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
