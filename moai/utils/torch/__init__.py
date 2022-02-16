import torch
import toolz
import typing

__all__ = ['get_submodule']

def get_submodule_pt_ge_110(
    module: torch.nn.Module,
    name: str
) -> torch.nn.Module:
    return module.get_submodule(name)

def get_submodule_pt_lt_110(
    module: torch.nn.Module,
    name: str
) -> torch.nn.Module:
    split = name.split('.')
    def _getattr(object: typing.Any, key: str):
        return getattr(object, key, None)
    return toolz.reduce(_getattr, split, module)

if torch.__version__ >= (1, 10):
    get_submodule = get_submodule_pt_ge_110
else:
    get_submodule = get_submodule_pt_lt_110