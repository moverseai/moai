from collections.abc import Callable

import torch
import typing
import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

__all__ = ['NoOp']

class NoOp(Callable):
    def __init__(self, 
        calculators: omegaconf.DictConfig=None,
    ):
        log.info("No data statistics used.")

    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:        
        pass
        