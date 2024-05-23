import moai.utils.engine as mieng

import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

__all__ = ["Exporter"]

class Exporter(mieng.Interval, mieng.Single):
    def __init__(self,
        batch_interval:         int,
        exporters:              omegaconf.DictConfig,        
    ):
        mieng.Interval.__init__(self, batch_interval)
        mieng.Single.__init__(self, items=exporters, name="exporter")
        log.info(f"Exporting data at a {batch_interval} step interval.")