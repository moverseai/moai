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
        self.milestones = OrderedDict(sorted(
            toolz.keymap(int, omegaconf.OmegaConf.to_container(milestones)).items()
        ))
        desc = pprint.pformat(dict(toolz.keymap(lambda x: f"epoch_{x}",omegaconf.OmegaConf.to_container(milestones)).items()))
        log.info(f"A monad parameter schedule is used:\n{desc}.")
            
    def setup(self, trainer, pl_module: FeedForward, stage: typing.Optional[str]=None) -> None:        
        keys = list(toolz.unique(toolz.concat(v.keys() for v in self.milestones.values())))
        for k in keys: 
            key_fields = toolz.merge(toolz.keyfilter(lambda d: k in d, v)
                for v in self.milestones.values()
            )
            for f in key_fields[k].keys():
                m = get_submodule(pl_module, k)
                if not hasattr(m, f):
                    log.warning(f"Monad [{m}] does not contain a [{f}] field, will be ignored when scheduling.")
                
    def on_train_epoch_start(self, trainer, pl_module: FeedForward):
        next = int(toolz.peek(self.milestones)[0]) if len(self.milestones) else sys.maxsize        
        if trainer.current_epoch + 1 >= next:
            ms = self.milestones.pop(next)
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