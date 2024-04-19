from moai.engine.modules.seed import RandomSeed

import hydra.utils as hyu
import omegaconf.omegaconf

__all__ = [
    "Engine",  
]

class Engine(object):
    def __init__(self,
        modules: omegaconf.DictConfig=None
    ):        
        # self.inits = [hyu.instantiate(conf) for conf in modules.values()]\
        #     if modules is not None else []
        if modules is None or 'manual_seed' not in modules.keys():
            RandomSeed()