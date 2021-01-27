import moai.utils.engine as mieng

import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

__all__ = ["Exporters"]

class Exporters(mieng.Collection, mieng.Interval):
    def __init__(self,
        batch_interval:         int,
        exporters:              omegaconf.DictConfig,        
    ):
        mieng.Interval.__init__(self, batch_interval)
        mieng.Collection.__init__(self, 
            items=exporters, name="exporters"
        )
        log.info(f"Exporting data at a {batch_interval} step interval.")

