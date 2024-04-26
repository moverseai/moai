from moai.monads.execution.cascade import (
    _create_accessor,
)
import numpy as np
import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import inspect
import typing
import itertools
import toolz
import logging
# import functools

log = logging.getLogger(__name__)

#TODO: use class param execs

__all__ = ["Models", "Tensors", "Criteria"]

from moai.export.local.pkl import TensorMonitor #TODO: use this as base class

class Criteria():
    __PRESET_ARGS__ = set(['step', 'epoch', 'batch_idx', 'optimization_step', 'stage', 'iter'])
    
    @classmethod
    def _dict_of_lists_to_list_of_dicts(_, DL): # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
        return [dict(zip(DL,t)) for t in zip(*DL.values())]
    
    class ArgsOperation():
        def __init__(self,
            func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
            tensor_args: typing.Mapping[str, str],
            other_args: typing.Mapping[str, str],
            kwargs: typing.Set[str],
        ):
            self.func = func
            #NOTE: DL to LD needs to ensure lists
            tensor_args = toolz.valmap(lambda v: [_create_accessor(a) for a in v], tensor_args)
            self.tensor_args = Criteria._dict_of_lists_to_list_of_dicts(tensor_args)
            self.other_args = Criteria._dict_of_lists_to_list_of_dicts(other_args)
            self.kwargs = kwargs

        def __call__(self, 
            tensors: typing.Mapping[str, torch.Tensor],
            meta: typing.Mapping[str, int]
        ) -> None:
            stop = False
            for i, args in enumerate(self.tensor_args):
                kwargs = toolz.valmap(
                    lambda a: a(tensors).detach().cpu().numpy().squeeze(), 
                    args
                )
                kwargs.update(toolz.get(i, self.other_args, {}))
                kwargs.update({k: meta[k] for k in self.kwargs if k in meta})
                stop = self.func(**kwargs)
                if stop:
                    break
            return stop

    class DictOperation():
        def __init__(self,
            func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
            args: typing.Mapping[str, str],
            kwargs: typing.Set[str],
        ):
            self.func = func
            self.args = Criteria._dict_of_lists_to_list_of_dicts(args)
            self.kwargs = kwargs

        def __call__(self, 
            tensors: typing.Mapping[str, torch.Tensor],
            meta: typing.Mapping[str, int]
        ) -> None:
            stop = False
            kwargs = {k: meta[k] for k in self.kwargs}
            for args in self.args or [{}]:
                stop = self.func(tensors, **args, **kwargs)
                if stop:
                    break
            return stop

    def is_first_arg_tensor_dict(self, arg: inspect.Parameter) -> bool:
        return typing.get_args(arg.annotation) == (str, torch.Tensor)

    def _get_tensor_args(self, 
        args: typing.Mapping[str, inspect.Parameter]
    ) -> typing.Mapping[str, inspect.Parameter]: #NOTE: do we need tensor args? or just numpy?
        def is_annotation_array_or_tensor(p: inspect.Parameter):
            return p.annotation == np.ndarray or p.annotation == torch.Tensor
        return toolz.keyfilter(lambda k: is_annotation_array_or_tensor(args[k]), args)

    def __init__(self, 
        criteria: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ) -> None:
        self.operations = []
        for k in kwargs or {}:
            params = kwargs[k] #NOTE: list args for multi calling
            override_params = params.get('params', None) or {}            
            target = criteria[k]
            operation = hyu.instantiate(target, **override_params)
            signature = inspect.signature(operation)
            extras = Criteria.__PRESET_ARGS__.intersection(signature.parameters.keys())            
            #NOTE: operate should change to partial func call
            if self.is_first_arg_tensor_dict(next(iter(signature.parameters.values()))):# signature.parameters[next(iter(args))]):
                args = set(toolz.drop(1, signature.parameters.keys())) - Criteria.__PRESET_ARGS__
                op_args = toolz.keyfilter(lambda k: k in args, params) #NOTE: currently assumes that no arg is named `params`
                self.operations.append(Criteria.DictOperation(operation, op_args, extras))
            else:
                args = signature.parameters.keys() - Criteria.__PRESET_ARGS__
                op_args = toolz.keyfilter(lambda k: k in args, params) #NOTE: currently assumes that no arg is named `params`
                tensor_args = self._get_tensor_args(signature.parameters)
                self.operations.append(Criteria.ArgsOperation(operation, 
                    toolz.dissoc(op_args,*(args-tensor_args.keys())), 
                    toolz.dissoc(op_args,*tensor_args.keys()),
                    extras))

    def __call__(self, 
        tensors:    typing.Mapping[str, torch.Tensor], 
        extras:     typing.Mapping[str, typing.Any],
    ) -> None:
        stop = False
        for operation in self.operations:
            stop = operation(tensors, extras)
            if stop:
                break
        return stop

class Tensors():
    __PRESET_ARGS__ = set(['step', 'epoch', 'batch_idx', 'optimization_step', 'stage', 'iter'])
    
    @classmethod
    def _dict_of_lists_to_list_of_dicts(_, DL): # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
        return [dict(zip(DL,t)) for t in zip(*DL.values())]
    
    class ArgsOperation():
        def __init__(self,
            func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
            tensor_args: typing.Mapping[str, str],
            other_args: typing.Mapping[str, str],
            kwargs: typing.Set[str],
        ):
            self.func = func
            #NOTE: DL to LD needs to ensure lists
            tensor_args = toolz.valmap(lambda v: [_create_accessor(a) for a in v], tensor_args)
            self.tensor_args = Tensors._dict_of_lists_to_list_of_dicts(tensor_args)
            self.other_args = Tensors._dict_of_lists_to_list_of_dicts(other_args)
            self.kwargs = kwargs

        def __call__(self, 
            tensors: typing.Mapping[str, torch.Tensor],
            meta: typing.Mapping[str, int]
        ) -> None:
            for i, args in enumerate(self.tensor_args):
                kwargs = toolz.valmap(
                    lambda a: a(tensors).detach().cpu().numpy().squeeze(), 
                    args
                )
                kwargs.update(toolz.get(i, self.other_args, {}))
                kwargs.update({k: meta[k] for k in self.kwargs if k in meta})
                self.func(**kwargs)

    class DictOperation():
        def __init__(self,
            func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
            args: typing.Mapping[str, str],
            kwargs: typing.Set[str],
        ):
            self.func = func
            self.args = Tensors._dict_of_lists_to_list_of_dicts(args)
            self.kwargs = kwargs

        def __call__(self, 
            tensors: typing.Mapping[str, torch.Tensor],
            meta: typing.Mapping[str, int]
        ) -> None:
            kwargs = {k: meta[k] for k in self.kwargs}
            for args in self.args or [{}]:
                self.func(tensors, **args, **kwargs)

    def is_first_arg_tensor_dict(self, arg: inspect.Parameter) -> bool:
        return typing.get_args(arg.annotation) == (str, torch.Tensor)

    def _get_tensor_args(self, 
        args: typing.Mapping[str, inspect.Parameter]
    ) -> typing.Mapping[str, inspect.Parameter]: #NOTE: do we need tensor args? or just numpy?
        def is_annotation_array_or_tensor(p: inspect.Parameter):
            return p.annotation == np.ndarray or p.annotation == torch.Tensor
        return toolz.keyfilter(lambda k: is_annotation_array_or_tensor(args[k]), args)

    def __init__(self, 
        tensors: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ) -> None:
        self.operations = []
        for k in kwargs or {}:
            params = kwargs[k] #NOTE: list args for multi calling
            override_params = params.get('params', None) or {}            
            target = tensors[k]
            operation = hyu.instantiate(target, **override_params)
            signature = inspect.signature(operation)
            extras = Tensors.__PRESET_ARGS__.intersection(signature.parameters.keys())            
            #NOTE: operate should change to partial func call
            if self.is_first_arg_tensor_dict(next(iter(signature.parameters.values()))):# signature.parameters[next(iter(args))]):
                args = set(toolz.drop(1, signature.parameters.keys())) - Tensors.__PRESET_ARGS__
                op_args = toolz.keyfilter(lambda k: k in args, params) #NOTE: currently assumes that no arg is named `params`
                self.operations.append(Tensors.DictOperation(operation, op_args, extras))
            else:
                args = signature.parameters.keys() - Tensors.__PRESET_ARGS__
                op_args = toolz.keyfilter(lambda k: k in args, params) #NOTE: currently assumes that no arg is named `params`
                tensor_args = self._get_tensor_args(signature.parameters)
                self.operations.append(Tensors.ArgsOperation(operation, 
                    toolz.dissoc(op_args,*(args-tensor_args.keys())), 
                    toolz.dissoc(op_args,*tensor_args.keys()),
                    extras))

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
