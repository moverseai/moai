import torch
import hydra.utils as hyu
import omegaconf.omegaconf
import inspect
import typing
import itertools
import toolz
import logging

log = logging.getLogger(__name__)

__all__ = ['Metrics']

from moai.monads.execution.cascade import _create_accessor

class Metrics(torch.nn.ModuleDict):
    execs: typing.List[typing.Callable] = []

    def __init__(self, 
        metrics: omegaconf.DictConfig={},
        **kwargs: typing.Mapping[str, typing.Any]
    ):
        super(Metrics, self).__init__()
        self.reductions = []
        if not len(metrics):
            log.warning("A collection of metrics is being used for validating the model, but no metrics have been assigned")
        loop = ((key, params) for key, params in kwargs.items() if key in metrics)
        #TODO: add message in case key is not found
        for k, p in loop:
            self.add_module(k, hyu.instantiate(getattr(metrics, k)))
            # last_module = toolz.last(self.modules()) # moduledict is ordered            
            last_module = self[k]
            sig = inspect.signature(last_module.forward)
            if 'out' not in p:
                p['out'] = [k]
            # if 'reduction' in p:
            #     reduction = iter(p['reduction'])
            # else:
            #     log.warning(f"{k} metric has no assigned reduction, automatically reverting to mean reduction.")
            #     reduction = itertools.cycle(['mean'])  
            for keys in zip(*toolz.remove(lambda x: not x, list(p[prop] for prop in itertools.chain(sig.parameters, ['out']) if p.get(prop) is not None))):
                accessors = [_create_accessor(k if isinstance(k, str) else toolz.get(0, k, None)) for k in keys[:-1]]
                self.execs.append(lambda tensor_dict, metric_dict,
                    acc=accessors, k=keys, p=sig.parameters.keys(), f=last_module:
                    metric_dict.update({
                        k[-1]: f(**dict(zip(p, 
                            list(
                                a(tensor_dict) for a, i in zip(acc, k[:-1])
                                if i is not None or None
                            )
                        )))
                    })
                )
                # self.reductions.append(next(reduction))

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        metrics = { }                
        for exe in self.execs:
            exe(tensors, metrics)
        returned = { }
        for i, (k, m) in enumerate(metrics.items()):
            if torch.is_tensor(m):
                returned[f'{k}'] = m
            else:
                log.warning(f"Metric [{k}] return type ({type(m)} is not supported and is being ignored.")                
        return returned