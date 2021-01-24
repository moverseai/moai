from collections.abc import Callable

import torch
import functools
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["PredictionBias"]

class PredictionBias(Callable):
    def __init__(self, 
        bias:           float=1.0,
        kernel_size:    int=3,
        out_channels:   int=1,
    ):
        self.bias = bias
        self.out_channels = out_channels
        self.size = (kernel_size, kernel_size)

    def __call__(self,
        module: torch.nn.Module
    ) -> None:        
        if isinstance(module, torch.nn.Conv2d):#TODO: other conv types?
            if module.out_channels == self.out_channels and module.kernel_size == self.size:
                for k, v in module.named_parameters():
                    if k == 'bias':
                        log.info(f"Bias initialization for {module} to {self.bias}.")
                        v.data.fill_(self.bias)
