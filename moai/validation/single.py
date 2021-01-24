import moai.utils.engine as mieng

import torch
import omegaconf.omegaconf
import typing
import logging
import inspect
import itertools

log = logging.getLogger(__name__)

__all__ = ['Metric']

class Metric(mieng.Single):
    def __init__(self,
        metrics: omegaconf.DictConfig,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super(Metric, self).__init__(
            items=metrics, 
            name="metric",
        )
        loop = ((key, params) for key, params in kwargs.items() if hasattr(indicators, key))
        for k, p in loop:
            last_module = self.metric
            sig = inspect.signature(last_module.forward)
            for keys in zip(*list(p[prop] for prop in itertools.chain(sig.parameters, ['out']))):
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
            returned[k] = torch.mean(m) if len(m.size()) > 0 else m                
        return returned