import moai.utils.engine as mieng

import torch
import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ['Optimizer']

class Optimizer(mieng.Single):
    def __init__(self,        
        parameters: typing.Iterator[torch.nn.Parameter],
        optimizers: omegaconf.DictConfig,
    ):
        super(Optimizer, self).__init__(
            items=optimizers, 
            name="optimizers",
            arguments=[parameters]
        )