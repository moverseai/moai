from moai.utils.torch import get_submodule

import torch
import typing
import toolz

__all__ = ['MonadParameterSelector']

class MonadParameterSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __init__(self,
        keys:       typing.Sequence[str],
    ):
        self.keys = keys        

    def __call__(self, module: torch.nn.Module) -> typing.List[torch.Tensor]:
        params = []
        for key in self.keys:
            # split = key.split('.')
            # m = toolz.reduce(getattr, split, module)
            m = get_submodule(module, key)
            params.append(m.parameters())
        # return list(toolz.concat(params))
        return { 'params': list(toolz.concat(params)) }
