from moai.utils.torch import get_submodule

import torch
import typing
import logging

__all__ = ['NamedParameterSelector']

log = logging.getLogger(__name__)

class NamedParameterSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __init__(self,
        keys:       typing.Sequence[str],
        force_grad: bool=True,
    ):
        self.keys = keys
        self.force_grad = force_grad

    def __call__(self, module: torch.nn.Module) -> typing.List[torch.Tensor]:
        params = []
        for k in self.keys:
            m = module
            found = True
            for s in k.split('.'): #TODO: check get_submodule in pytorch 1.10
                found = found and hasattr(m, s)
                if found:
                    m = getattr(m, s)
                else:
                    break
            if found:
                if self.force_grad:
                    m.requires_grad_(True)
                params.append(m)
            else:
                log.warning(f"Parameter {k} not found!")
        return params