import moai.utils.engine as mieng

import torch
import omegaconf.omegaconf
import logging 

log = logging.getLogger(__name__)

__all__ = ["Schedulers"]

class Schedulers(mieng.Collection):
    def __init__(self, 
        schedulers: omegaconf.DictConfig
    ):
        super(Schedulers, self).__init__(items=schedulers, name="schedulers")

