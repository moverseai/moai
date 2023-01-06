import moai.utils.engine as mieng

import torch
import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

__all__ = ["Initializer"]

class Initializer(mieng.Single):
    def __init__(self, 
        schemes: omegaconf.DictConfig
    ):
        super(Initializer, self).__init__(items=schemes, name="scheme")

    def __call__(self, 
        model: torch.nn.Module
    ) -> None:
        count = len(self.items())
        if count:
            log.info(f"Applying {count} parameter initialization schemes:")
        for scheme in self.items():
            log.info(f"\tApplying {scheme.__class__.__name__}")
            model.apply(scheme)