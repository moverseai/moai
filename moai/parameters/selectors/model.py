import torch
import typing

__all__ = ['ModelParameterSelector']

class ModelParameterSelector(typing.Callable[[torch.nn.Module], typing.List[torch.Tensor]]):
    def __init__(self,
        keys:       typing.Sequence[str],
    ):
        self.keys = keys        

    def __call__(self, module: torch.nn.Module) -> typing.List[torch.Tensor]:
        params = []
        for key in self.keys:
            t = module.predictions[key]
            t.requires_grad_(True)
            params.append(t)
        return { 'params': params }
        # return params