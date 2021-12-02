import torch
import toolz
import typing
import logging

log = logging.getLogger(__name__)

__all__ = ["ZeroParams"]

class ZeroParams(typing.Callable[[torch.nn.Module], None]):
    def __init__(self, 
        keys:           typing.Sequence[str],
    ):
        self.keys = keys
        
    def __call__(self,
        module: torch.nn.Module
    ) -> None:        
        for key in self.keys:
            try:
                # m = module.get_submodule(key) #NOTE: #TODO: for PyTorch 1.10
                split = key.split('.')
                def _getattr(object: typing.Any, key: str):
                    return getattr(object, key, None)
                m = toolz.reduce(_getattr, split, module)
                if m is not None:
                    log.info(f"Zeroing out parameter: {key}.")
                    with torch.no_grad():
                        m.zero_()
            except:
                break
            
