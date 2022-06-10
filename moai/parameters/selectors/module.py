from moai.utils.torch import get_submodule

import torch
import typing
import toolz

__all__ = ['ModuleParameterSelector']

class ModuleParameterSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __init__(self,
        keys:       typing.Sequence[str],
        force_grad: bool=True,
    ):
        self.keys = keys
        self.force_grad = force_grad

    def __call__(self, module: torch.nn.Module) -> typing.List[torch.Tensor]:
        params = []
        for key in self.keys: #TODO: check get_submodule in pytorch 1.10
            # split = key.split('.')
            # m = toolz.reduce(getattr, split, module)
            m = get_submodule(module, key)
            params.append(m.parameters())
        parameters = list(toolz.concat(params))
        if self.force_grad:
            for p in parameters:            
                p.requires_grad_(True)
        return { 'params': parameters }
        # return parameters