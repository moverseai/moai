import moai.utils.engine as mieng

import omegaconf.omegaconf
import logging

log = logging.getLogger(__name__)

__all__ = ["Latent_Visualizers"]

class LatentVisualizers(mieng.Collection, mieng.Interval):
    def __init__(self,
        batch_interval:int,
        visualizers: omegaconf.DictConfig,
        latent_visualizers: omegaconf.DictConfig,
    ):
        mieng.Interval.__init__(self, batch_interval)
        mieng.Collection.__init__(
            self, 
            items=latent_visualizers or {}, 
            name="latent_visualizers"
        )
        mieng.Collection.__init__(
            self, 
            items=visualizers, 
            name="visualizers"
        )