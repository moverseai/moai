import numpy as np
import typing
import toolz
import torch
import inspect
import functools

__PRESET_ARGS__ = set(['step', 'epoch', 'batch_idx', 'optimization_step', 'stage', 'iter'])
    
def _mul(
    tensor_dict: typing.Dict[str, torch.Tensor],
    keys: typing.Sequence[str],
) -> torch.Tensor:
    return tensor_dict[keys[0]] * tensor_dict[keys[1]]

def _add(
    tensor_dict: typing.Dict[str, torch.Tensor],
    keys: typing.Sequence[str],
) -> torch.Tensor:
    return tensor_dict[keys[0]] + tensor_dict[keys[1]]

def _sub(
    tensor_dict: typing.Dict[str, torch.Tensor],
    keys: typing.Sequence[str],
) -> torch.Tensor:
    return tensor_dict[keys[0]] - tensor_dict[keys[1]]

def _div(
    tensor_dict: typing.Dict[str, torch.Tensor],
    keys: typing.Sequence[str],
) -> torch.Tensor:
    return tensor_dict[keys[0]] / tensor_dict[keys[1]]

def _cat(
    tensor_dict: typing.Dict[str, torch.Tensor],
    keys: typing.Sequence[str],
) -> torch.Tensor:
    return torch.cat([tensor_dict[keys[0]], tensor_dict[keys[1]]], dim=1)

def _dict(
    tensor_dict: typing.Dict[str, torch.Tensor],
    keys: typing.Sequence[str],
) -> torch.Tensor:
    return toolz.get_in(keys, tensor_dict, no_default=True) #NOTE: should crash if the key is not found

__ACCESSORS__ = {
    ' * ': _mul,
    ' + ': _add,
    ' - ': _sub,
    ' / ': _div,
    ' | ': _cat,
    '.': _dict,
}

#TODO: need a lexer/parser/grammar here...
def _create_accessor(key: typing.Optional[typing.Union[str, typing.Sequence[str]]]):
    if key is None:
        return lambda _: None
    for k in __ACCESSORS__.keys():
        if k in key:
            return functools.partial(__ACCESSORS__[k], keys=key.split(k))
    return lambda td, k=key: td[k]

def _dict_of_lists_to_list_of_dicts(DL): # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    return [dict(zip(DL,t)) for t in zip(*DL.values())]
           
def _is_first_arg_tensor_dict(arg: inspect.Parameter) -> bool:
    return typing.get_args(arg.annotation) == (str, torch.Tensor)

def _get_tensor_args( 
    args: typing.Mapping[str, inspect.Parameter]
) -> typing.Mapping[str, inspect.Parameter]: #NOTE: do we need tensor args? or just numpy?
    def is_annotation_array_or_tensor(p: inspect.Parameter):
        return p.annotation == np.ndarray or p.annotation == torch.Tensor
    return toolz.keyfilter(lambda k: is_annotation_array_or_tensor(args[k]), args)

class CriteriaArgsOperation():
    def __init__(self,
        func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
        tensor_args: typing.Mapping[str, str],
        other_args: typing.Mapping[str, str],
        kwargs: typing.Set[str],
    ):
        self.func = func
        #NOTE: DL to LD needs to ensure lists
        tensor_args = toolz.valmap(lambda v: [_create_accessor(a) for a in v], tensor_args)
        self.tensor_args = _dict_of_lists_to_list_of_dicts(tensor_args)
        self.other_args = _dict_of_lists_to_list_of_dicts(other_args)
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

class CriteriaDictOperation():
    def __init__(self,
        func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
        args: typing.Mapping[str, str],
        kwargs: typing.Set[str],
    ):
        self.func = func
        self.args = _dict_of_lists_to_list_of_dicts(args)
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

class TensorsArgsOperation():
    def __init__(self,
        func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
        tensor_args: typing.Mapping[str, str],
        other_args: typing.Mapping[str, str],
        kwargs: typing.Set[str],
    ):
        self.func = func
        #NOTE: DL to LD needs to ensure lists
        tensor_args = toolz.valmap(lambda v: [_create_accessor(a) for a in v], tensor_args)
        self.tensor_args = _dict_of_lists_to_list_of_dicts(tensor_args)
        self.other_args = _dict_of_lists_to_list_of_dicts(other_args)
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

class TensorsDictOperation():
    def __init__(self,
        func: typing.Callable[[typing.Union[torch.Tensor, typing.Sequence[torch.Tensor], typing.Mapping[str, torch.Tensor]]], None],
        args: typing.Mapping[str, str],
        kwargs: typing.Set[str],
    ):
        self.func = func
        self.args = _dict_of_lists_to_list_of_dicts(args)
        self.kwargs = kwargs

    def __call__(self, 
        tensors: typing.Mapping[str, torch.Tensor],
        meta: typing.Mapping[str, int]
    ) -> None:
        kwargs = {k: meta[k] for k in self.kwargs}
        for args in self.args or [{}]:
            self.func(tensors, **args, **kwargs)