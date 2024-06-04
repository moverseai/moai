import typing

import toolz
import torch

__all__ = [
    "get_submodule",
    "cross_product",
]


def get_submodule_pt_ge_110(module: torch.nn.Module, name: str) -> torch.nn.Module:
    return module.get_submodule(name)


def get_parameter_pt_ge_110(module: torch.nn.Module, name: str) -> torch.nn.Module:
    return module.get_parameter(name)


def get_child_pt_lt_110(module: torch.nn.Module, name: str) -> torch.nn.Module:
    split = name.split(".")

    def _getattr(object: typing.Any, key: str):
        return getattr(object, key, None)

    return toolz.reduce(_getattr, split, module)


if isinstance(torch.__version__, str):
    v = torch.__version__.split(".")
else:
    v = torch.__version__

if (int(v[0]), int(v[1])) >= (1, 10):
    get_submodule = get_submodule_pt_ge_110
else:
    get_submodule = get_child_pt_lt_110

if (int(v[0]), int(v[1])) >= (1, 10):
    get_parameter = get_parameter_pt_ge_110
else:
    get_parameter = get_child_pt_lt_110

if (int(v[0]), int(v[1])) >= (1, 12):
    cross_product = torch.linalg.cross
else:
    cross_product = torch.cross
