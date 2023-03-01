from moai.utils.torch import get_submodule
from moai.networks.lightning import FeedForward
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict

import dataclasses
import pprint
import omegaconf
import sys
import toolz
import logging
import typing

log = logging.getLogger(__name__)

__all__ = ["MonadSchedule"]

class MonadSchedule(Callback):
    def __init__(self,        
        milestones: omegaconf.DictConfig,        
    ):
        super().__init__()
        self.milestones = {
            'epoch': OrderedDict(sorted(toolz.keymap(int, 
                    omegaconf.OmegaConf.to_container(milestones.epoch)\
                        if milestones.epoch else {}
                ).items()
            )),
            'step': OrderedDict(sorted(toolz.keymap(int, 
                    omegaconf.OmegaConf.to_container(milestones.step)\
                        if milestones.step else {}
                ).items()
            ))
        } 
        desc = pprint.pformat(omegaconf.OmegaConf.to_container(milestones))
        log.info(f"A monad parameter schedule is used:\n{desc}.")
            
    def setup(self, trainer, pl_module: FeedForward, stage: typing.Optional[str]=None) -> None:        
        for key, data in self.milestones.items():
            keys = list(toolz.unique(toolz.concat(v.keys() for v in data.values())))
            for k in keys: 
                key_fields = toolz.merge(toolz.keyfilter(lambda d: k in d, v)
                    for v in data.values()
                )
                for f in key_fields[k].keys():
                    m = get_submodule(pl_module, k)
                    if not hasattr(m, f):
                        log.warning(f"Monad [{m}] does not contain a [{f}] field, will be ignored when scheduling.")
                
    def on_train_epoch_start(self, trainer, pl_module: FeedForward):        
        next = int(toolz.peek(self.milestones['epoch'])[0])\
            if len(self.milestones['epoch']) else sys.maxsize        
        if trainer.current_epoch + 1 >= next:
            ms = self.milestones['epoch'].pop(next)
            for m, vs in ms.items():
                module = get_submodule(pl_module, m)
                # if not dataclasses.is_dataclass(module):
                #     log.warning(f"Monad [{m}] is not a `dataclass` and will be ignored.")
                #     continue                
                for k, v in vs.items():
                    if not hasattr(module, k):
                        log.info(f"Ignoring field [{k}] of [{m}] monad.")
                    else:
                        setattr(module, k, v)
                        log.info(f"Updated [{m}] parameter [{k}] to [{v}] @ epoch {next}.")

    def on_train_batch_start(self, 
        trainer, pl_module: FeedForward, 
        batch: typing.Any, batch_idx: int
    ):
        next = int(toolz.peek(self.milestones['step'])[0])\
            if len(self.milestones['step']) else sys.maxsize        
        if pl_module.global_step + 1 >= next:
            ms = self.milestones['step'].pop(next)
            for m, vs in ms.items():
                module = get_submodule(pl_module, m)
                # if not dataclasses.is_dataclass(module):
                #     log.warning(f"Monad [{m}] is not a `dataclass` and will be ignored.")
                #     continue                
                for k, v in vs.items():
                    if not hasattr(module, k):
                        log.info(f"Ignoring field [{k}] of [{m}] monad.")
                    else:
                        setattr(module, k, v)
                        log.info(f"Updated [{m}] parameter [{k}] to [{v}] @ step#{next}.")