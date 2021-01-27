import moai.utils.engine as mieng

import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

__all__ = ["Visualizer"]

class Visualizer(mieng.Interval, mieng.Single):
    def __init__(self,
        batch_interval:int,
        visualizers: omegaconf.DictConfig,        
    ):
        mieng.Interval.__init__(self, batch_interval)
        mieng.Single.__init__(self, items=visualizers)