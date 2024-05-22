from moai.utils.torch import get_submodule

import torch
import typing

__all__ = ['SupervisionParameterSelector']

class SupervisionParameterSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __call__(self, module: torch.nn.Module) -> typing.List[torch.Tensor]:
        # return list(get_submodule(module, 'supervision').parameters())
        return { 'params': list(get_submodule(module, 'supervision').parameters()) }