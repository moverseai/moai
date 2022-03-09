from pytorch_lightning.callbacks import Callback
import moai.utils.engine as mieng

import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

__all__ = ["Visualizers"]

class Visualizers(mieng.Collection, mieng.Interval):
    def __init__(self,
        batch_interval:int,
        visualizers: omegaconf.DictConfig,
    ):
        mieng.Interval.__init__(self, batch_interval)
        mieng.Collection.__init__(
            self, 
            items=visualizers, 
            name="visualizers"
        )