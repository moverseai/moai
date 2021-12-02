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

__all__ = ["Cascade"]

def _mul(
    tensor_dict: typing.Dict[str, torch.Tensor],
    lhs_key: str, rhs_key: str
) -> torch.Tensor:
    return tensor_dict[lhs_key] * tensor_dict[rhs_key]

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
    return toolz.get_in(keys, tensor_dict)

__ACCESSORS__ = {
    ' * ': _mul,
    ' + ': _add,
    ' - ': _sub,
    ' / ': _div,
    ' | ': _cat,
    '.': _dict,
}

def _create_accessor(key: typing.Union[str, typing.Sequence[str]]):
    for k in __ACCESSORS__.keys():
        if k in key:
            return functools.partial(__ACCESSORS__[k], keys=key.split(k))
    return lambda td, k=key: td[k]


class Cascade(torch.nn.ModuleDict): #TODO: check if x: ['arg'] is the same as x: 'arg'
    execs: typing.List[typing.Callable]

    def __init__(self, 
        monads: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any]
    ):
        super(Cascade, self).__init__()
        #TODO: first construct via monads and then create lambdas via kwargs
        loop = ((key, params) for key, params in kwargs.items() if key in monads.keys())
        #TODO: add message in case key is not found
        self.execs = []
        for k, p in loop:
            #TODO: if instantiation fails, error out with message saying about adding it to the config file
            self.add_module(k, hyu.instantiate(getattr(monads, k))) #TODO: add construction loop first in class execs, and then funcs based on kwargs
            # last_module = toolz.last(self.modules()) #NOTE: moduledict is ordered            
            module = self[k]
            sig = inspect.signature(module.forward)
            props = [prop for prop in sig.parameters if p[prop] is not None]
            for keys in zip(*list(p[prop] for prop in itertools.chain(props, ['out']))):
                accessors = [_create_accessor(k if isinstance(k, str) else k[0]) for k in keys[:-1]]
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
