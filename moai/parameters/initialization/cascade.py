import moai.utils.engine as mieng

import torch
import omegaconf.omegaconf
import logging 

log = logging.getLogger(__name__)

__all__ = ["Initializers"]

class Initializers(mieng.Collection):
    def __init__(self, 
        schemes: omegaconf.DictConfig={}
    ):
        super(Initializers, self).__init__(items=schemes, name="schemes")

    def __call__(self, 
        model: torch.nn.Module
    ) -> None:
        log.info(f"Applying {len(self.items())} parameter initialization schemes:")
        for scheme in self.items():
            log.info(f"\tApplying {scheme.__class__.__name__}.")
            model.apply(scheme)

