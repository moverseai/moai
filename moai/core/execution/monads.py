import moai.core.execution.common as mic
import typing
import torch
import omegaconf.omegaconf
import hydra.utils as hyu
import inspect
import itertools
import logging

log = logging.getLogger(__name__)

__all__ = ['Monads']

class Monads(torch.nn.ModuleDict): #TODO: check if x: ['arg'] is the same as x: 'arg'
    execs: typing.List[typing.Callable]

    def __init__(self, 
        monads: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        #TODO: first construct via monads and then create lambdas via kwargs
        loop = ((key, params) for key, params in kwargs.items() if key in monads.keys())
        #NOTE: check for not found keys and notify the potential error
        errors = [k for k in kwargs if k not in monads]
        if errors:
            log.error(f"The following monads {errors} were not found in the configuration and will be ignored!")
        self.execs = []
        for k, p in loop:
            #TODO: if instantiation fails, error out with message saying about adding it to the config file
            try:
                self.add_module(k, hyu.instantiate(getattr(monads, k))) #TODO: add construction loop first in class execs, and then funcs based on kwargs
            except Exception as e:
                log.error(f"Could not instantiate the monad {k} with exception {e}!")
                continue
            # last_module = toolz.last(self.modules()) #NOTE: moduledict is ordered            
            module = self[k]
            sig = inspect.signature(module.forward)
            props = [prop for prop in sig.parameters if prop in p]
            for keys in zip(*list(p[prop] for prop in itertools.chain(props, ['out']))):
                accessors = [mic._create_accessor(k if isinstance(k, str) or k is None else k[0]) for k in keys[:-1]]
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