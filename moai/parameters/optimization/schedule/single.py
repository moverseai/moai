import moai.utils.engine as mieng

import torch.optim
import omegaconf.omegaconf
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["Scheduler"]

class Scheduler(mieng.Single):
    def __init__(self,
        optimizers: typing.Sequence[torch.optim.Optimizer],
        schedulers: omegaconf.DictConfig,        
    ):
        super(Scheduler, self).__init__(items=schedulers, 
            arguments=optimizers, name="schedulers"
        )