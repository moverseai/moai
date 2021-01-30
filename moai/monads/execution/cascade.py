import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import inspect
import typing
import itertools
import toolz
import logging

log = logging.getLogger(__name__)

#TODO: use class param execs

__all__ = ["Cascade"]

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
            last_module = toolz.last(self.modules()) #NOTE: moduledict is ordered            
            sig = inspect.signature(last_module.forward)
            props = [prop for prop in sig.parameters if p[prop] is not None]
            for keys in zip(*list(p[prop] for prop in itertools.chain(props, ['out']))):
                self.execs.append(lambda tensor_dict, k=keys, p=props, f=last_module:
                    tensor_dict.update({
                        k[-1]: f(**dict(zip(p, 
                            list(tensor_dict[i] if type(i) is str
                                else list(tensor_dict[j] for j in i) if i is not None else None
                                for i in k[:-1]
                            )
                        )))
                    })
                )                

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]    
    ) -> typing.Dict[str, torch.Tensor]:
        for exe in self.execs:
            exe(tensors) #NOTE: each executing lambda updates the tensor dict itself
        return tensors
