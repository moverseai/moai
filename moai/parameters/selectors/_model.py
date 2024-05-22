from moai.utils.torch import get_submodule

import torch
import typing
import toolz

__all__ = ['ModelParameterSelector']

class ModelParameterSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __init__(self,
        keys:       typing.Sequence[str],
        force_grad: bool=True,
    ):
        self.keys = keys
        self.force_grad = force_grad

    def __call__(self, module: torch.nn.Module) -> typing.List[torch.Tensor]:
        params = []
        # for key in self.keys:
        #     t = module.predictions[key]
        #     t.requires_grad_(True)
        #     params.append(t)
        for key in self.keys:
            # split = key.split('.')
            # m = toolz.reduce(getattr, split, module)
            m = get_submodule(module.models, key)
            params.append(m.parameters())
        parameters = list(toolz.concat(params))
        if self.force_grad:
            for p in parameters:            
                p.requires_grad_(True)
        return { 'params': parameters }
        # return params