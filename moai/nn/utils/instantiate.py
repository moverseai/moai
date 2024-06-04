import inspect
import typing

import torch

__all__ = ["instantiate"]


def instantiate(which_type: typing.Type, **kwargs: typing.Dict) -> torch.nn.Module:
    ctor_sig = inspect.signature(which_type)
    ctor_args, ctor_kwargs = {}, {}
    all_params = dict(kwargs)
    for k in ctor_sig.parameters.keys():
        if (
            k in all_params.keys()
            and ctor_sig.parameters[k].default == inspect._empty
            and (
                ctor_sig.parameters[k].kind
                == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                or ctor_sig.parameters[k].kind == inspect._ParameterKind.POSITIONAL_ONLY
            )
        ):
            ctor_args[k] = all_params[k]
        if (
            k in all_params.keys()
            and ctor_sig.parameters[k].default != inspect._empty
            and (
                ctor_sig.parameters[k].kind
                == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                or ctor_sig.parameters[k].kind == inspect._ParameterKind.KEYWORD_ONLY
            )
        ):
            ctor_kwargs[k] = all_params[k]
    bound_args = ctor_sig.bind(*ctor_args.values(), **ctor_kwargs)
    return which_type(*bound_args.args, **bound_args.kwargs)
