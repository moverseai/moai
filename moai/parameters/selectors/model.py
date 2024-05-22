from moai.utils.torch import get_submodule

import torch
import typing
import toolz

__all__ = ['ModelParameterSelector']

class ModelParameterSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __init__(self,
        modules:       typing.Sequence[str]=[],
        monads:        typing.Sequence[str]=[],
        parameters:    typing.Sequence[str]=[],
        force_grad: bool=True,
    ):
        self.modules = modules
        self.monads = monads
        self.parameters = parameters
        self.force_grad = force_grad

    def __call__(self, moai_model: torch.nn.Module) -> typing.List[torch.Tensor]:
        params = []
        for key in self.modules:
            m = get_submodule(moai_model.models, key)
            params.append(m.parameters())
        for key in self.monads:
            m = get_submodule(moai_model.named_flows, key)
            params.append(m.parameters())
        parameters = list(toolz.concat(params))
        for key in self.parameters:
            keys = key.split('.')
            m = get_submodule(moai_model.named_flows, ".".join(keys[:-1]))
            p = getattr(m, keys[-1])
            parameters.append(p)        
        if self.force_grad:
            for p in parameters:            
                p.requires_grad_(True)
        return { 'params': parameters }