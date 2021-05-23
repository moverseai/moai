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

class Metrics(torch.nn.ModuleDict):
    execs: typing.List[typing.Callable] = []

    def __init__(self, 
        metrics: omegaconf.DictConfig={},
        **kwargs: typing.Mapping[str, typing.Any]
    ):
        super(Metrics, self).__init__()
        if not len(metrics):
            log.warning("A collection of metrics is being used for validating the model, but no metrics have been assigned")
        loop = ((key, params) for key, params in kwargs.items() if key in metrics)
        #TODO: add message in case key is not found
        for k, p in loop:
            self.add_module(k, hyu.instantiate(getattr(metrics, k)))
            last_module = toolz.last(self.modules()) # moduledict is ordered            
            sig = inspect.signature(last_module.forward)
            if 'out' not in p:
                p['out'] = [k]
            for keys in zip(*toolz.remove(lambda x: not x, list(p[prop] for prop in itertools.chain(sig.parameters, ['out']) if p.get(prop) is not None))):
                self.execs.append(lambda tensor_dict, metric_dict, k=keys, p=sig.parameters.keys(), f=last_module:
                    metric_dict.update({
                        k[-1]: f(**dict(zip(p, 
                            list(tensor_dict[i] for i in k[:-1])
                        )))
                    })
                )

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        metrics = { }
        for exe in self.execs:
            exe(tensors, metrics)
        returned = { }
        for k, m in metrics.items():
            returned[f'{k}'] = m
        return returned