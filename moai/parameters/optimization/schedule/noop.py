from moai.parameters.optimization.schedule.schedulers import Identity

import torch
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["NoOp"]

class NoOp(object):
    def __init__(self,
        optimizers: typing.Sequence[torch.optim.Optimizer]
    ):
        self.schedulers = [Identity(next(iter(optimizers)))]

    def items(self):
        return self.schedulers