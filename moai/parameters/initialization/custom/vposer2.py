from moai.monads.human.body.prior.human_body_prior import VPoser_v2 as VPoser

import toolz
import torch
import logging
import typing
import os
import glob

log = logging.getLogger(__name__)

__all__ = ["VPoser2"]

class VPoser2(typing.Callable[[torch.nn.Module], None]):
    def __init__(self, 
        ckpt:       str,        
        cache:      bool=False,
    ):
        if os.path.isdir(ckpt):
            self.filename = glob.glob(os.path.join(ckpt, 'snapshots', '*.ckpt'))[-1]
        else:
            self.filename = ckpt
        if cache:
            self.state_dict = torch.load(
                self.filename, map_location=lambda storage, loc: storage
            )['state_dict']
        
    def __call__(self,
        module: torch.nn.Module
    ) -> None:        
        if isinstance(module, VPoser):
            state = self.state_dict if hasattr(self, 'state_dict') else torch.load(
                self.filename, map_location=lambda storage, loc: storage
            )['state_dict']
            module.load_state_dict(
                toolz.keymap(lambda k: k.replace('vp_model.', ''), state),
                strict=False
            )
            log.info(f"Initialized VPoser2 from {self.filename}")