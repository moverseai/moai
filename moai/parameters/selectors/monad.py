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
        for key in self.keys: #TODO: check get_submodule in pytorch 1.10
            split = key.split('.')
            m = toolz.reduce(getattr, split, module)
            params.append(m.parameters())
        return list(toolz.concat(params))