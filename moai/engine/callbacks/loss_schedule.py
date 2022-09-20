from moai.utils.torch import get_submodule
from moai.networks.lightning import FeedForward
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict

import pprint
import omegaconf.omegaconf
import sys
import toolz
import logging

log = logging.getLogger(__name__)

__all__ = ["LossSchedule"]

class LossSchedule(Callback):
    def __init__(self,        
        milestones: omegaconf.DictConfig,        
    ):
        super().__init__()
        self.milestones = OrderedDict(sorted(
            toolz.keymap(int, omegaconf.OmegaConf.to_container(milestones)).items()
        ))
        desc = pprint.pformat(dict(toolz.keymap(lambda x: f"epoch_{x}",omegaconf.OmegaConf.to_container(milestones)).items()))
        log.info(f"A weighting supervision schedule is used:\n{desc}.")

    def on_train_epoch_start(self, trainer, pl_module: FeedForward):
        next = int(toolz.peek(self.milestones)[0]) if len(self.milestones) else sys.maxsize
        supervision = get_submodule(pl_module, 'supervision')
        if trainer.current_epoch + 1 >= next:
            ms = self.milestones.pop(next)
            for k, v in ms.items():
                supervision.weights[k] = v
                log.info(f"Updated weight ({v}) for loss [{k}] @ epoch {next}.")