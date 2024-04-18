from moai.monads.execution.cascade import (
    _create_accessor,
)
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import inspect
import typing
import itertools
import toolz
import logging
import functools

log = logging.getLogger(__name__)

#TODO: use class param execs

__all__ = ["Models"]

from moai.export.local.pkl import TensorMonitor
    
class Tensors():
    __PRESET_ARGS__ = set(['step', 'epoch', 'batch_idx'])
    
    class ArgsOperation():
        def __init__(self,
            func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
            args: typing.Mapping[str, str],
            kwargs: typing.Set[str],
        ):
            self.func = func
            self.args = {k: _create_accessor(a) for k, a in args}
            self.kwargs = kwargs

        # def _access_tensors(self, 
        #     tensors: typing.Mapping[str, torch.Tensor],
        #     accessor: typing.Callable
        # ) -> torch.Tensor:
        #     return accessor(tensors)

        def __call__(self, 
            tensors: typing.Mapping[str, torch.Tensor],
            meta: typing.Mapping[str, int]
        ) -> None:
            # kwargs = toolz.valmap(self._access_tensors, self.args)
            kwargs = toolz.valmap(lambda a: a(tensors), self.args)
            kwargs.update({k: meta[k] for k in self.kwargs})
            self.func(**kwargs)

    class DictOperation():
        def __init__(self,
            func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
            kwargs: typing.Set[str],
        ):
            self.func = func
            self.kwargs = kwargs

        def __call__(self, 
            tensors: typing.Mapping[str, torch.Tensor],
            meta: typing.Mapping[str, int]
        ) -> None:
            kwargs = {k: meta[k] for k in self.kwargs}
            self.func(tensors, **kwargs)

    def is_first_arg_tensor_dict(self, arg: inspect.Parameter) -> bool:
        return typing.get_args(arg.annotation) == (str, torch.Tensor)

    def __init__(self, 
        tensors: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ) -> None:
        self.operations = []
        for k in kwargs or {}:
            override_params = kwargs[k] #NOTE: list args for multi calling
            target = tensors[k]
            operation = hyu.instantiate(target, **override_params)
            signature = inspect.signature(operation)
            extras = Tensors.__PRESET_ARGS__.intersection(signature.parameters.keys())
            args = signature.parameters.keys() - Tensors.__PRESET_ARGS__
            #NOTE: operate should change to partial func call
            if self.is_first_arg_tensor_dict(signature.parameters[next(iter(args))]):
                self.operations.append(Tensors.DictOperation(operation, extras))
            # else:
            #     op_args = 
            #     self.operations.append(Tensors.ArgsOperation(operation, *, extras))

    def __call__(self, 
        tensors:    typing.Mapping[str, torch.Tensor], 
        extras:     typing.Mapping[str, typing.Any],
    ) -> None:
        for operation in self.operations:
            operation(tensors, extras)

# inspect.signature(f).parameters['tensors'].annotation == typing.Dict[str, torch.Tensor]
# issubclass(type(pkl), typing.Callable)
# typing.get_args(inspect.signature(f).parameters['tensors'].annotation) == (str, torch.Tensor)

class Models(torch.nn.ModuleDict): #TODO: check if x: ['arg'] is the same as x: 'arg'
    execs: typing.List[typing.Callable]

    def __init__(self, 
        models: torch.nn.ModuleDict,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        #TODO: first construct via monads and then create lambdas via kwargs
        loop = ((key, params) for key, params in kwargs.items() if key in models.keys())
        #NOTE: check for not found keys and notify the potential error
        errors = [k for k in kwargs if k not in models]
        if errors:
            log.error(f"The following models {errors} were not found in the configuration and will be ignored!")
        self.execs = []
        for k, p in loop:            
            module = models[k]
            sig = inspect.signature(module.forward)
            props = [prop for prop in sig.parameters if p[prop] is not None]
            for keys in zip(*list(p[prop] for prop in itertools.chain(props, ['out']))):
                accessors = [_create_accessor(k if isinstance(k, str) or k is None else k[0]) for k in keys[:-1]]
                self.execs.append(lambda tensor_dict, 
                        acc=accessors, k=keys, p=props, f=module:
                    tensor_dict.update({
                        k[-1]: f(**dict(zip(p,
                            list(a(tensor_dict) if type(i) is str 
                                else list(tensor_dict[j] for j in i) if i is not None else None
                                for a, i in zip(acc, k[:-1])
                            )# list(tensor_dict[i] if type(i) is str
                        )))
                    })
                )

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for exe in self.execs:
            exe(tensors) #NOTE: each executing lambda updates the tensor dict itself
        return tensors
