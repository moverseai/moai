from collections.abc import Callable

import torch
import logging

log = logging.getLogger(__name__)

__all__ = ["Xavier"]

class Xavier(Callable):
    def __init__(self,
        zero_bias: bool,
    ):        
        self.conv_w_init = torch.nn.init.xavier_normal_
        self.w_init = torch.nn.init.normal_
        self.conv_b_init = torch.nn.init.zeros_ if zero_bias else torch.nn.init.normal_

    def __call__(self, 
        module: torch.nn.Module
    ) -> None:
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
            self.conv_w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose3d):
            self.conv_w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
        elif isinstance(module, torch.nn.BatchNorm1d):
            self.w_init(module.weight, mean=1, std=0.02)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.BatchNorm2d):
            self.w_init(module.weight, mean=1, std=0.02)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.BatchNorm3d):
            self.w_init(module.weight, mean=1, std=0.02)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.Linear):
            self.conv_w_init(module.weight)
            if module.bias is not None:
                self.conv_b_init(module.bias)
