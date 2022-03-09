from moai.parameters.initialization import Cascade
from moai.utils.torch import get_submodule

import torch
import omegaconf.omegaconf
import toolz
import logging

log = logging.getLogger(__name__)

__all__ = ["Named"]

class Named(Cascade):
    def __init__(self,         
        schemes: omegaconf.DictConfig={},
        modules: omegaconf.DictConfig={},        
    ):
        super(Named, self).__init__(schemes)
        self.inits = modules

    def __call__(self,
        module: torch.nn.Module
    ) -> None:
        found = []
        for k, v in self.inits.items():
            m = get_submodule(module, k)
            if m is not None:
                found += [k]
                log.info(f"Initializing {k} from {v.ckpt} " + ("strictly." if v.strict else "relaxed."))
                data = torch.load(v.ckpt, map_location=lambda s, l: s)
                if v.source:
                    split = v.source.split('.')
                    data = toolz.get_in(split, data) or data
                if v.replace:
                    for src, tgt in v.replace.items():
                        data = toolz.keymap(lambda k: k.replace(src, tgt), data)
                m.load_state_dict(data, strict=v.strict or False)
        for k in found:
            self.inits.pop(k)
        super(Named, self).__call__(module)
