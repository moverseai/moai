from moai.validation.collection import Metrics

import torch
import omegaconf.omegaconf
import typing
import logging
import toolz

log = logging.getLogger(__name__)

__all__ = ['Indicators']

class Indicators(Metrics):
    execs: typing.List[typing.Callable] = []

    def __init__(self, 
        metrics:            omegaconf.DictConfig={},
        indicators:         typing.Sequence[str]=[],
        **kwargs:           typing.Mapping[str, typing.Any]
    ):
        super(Indicators, self).__init__(metrics=metrics, **kwargs)
        self.expressions = { }
        for name, expression in indicators.items(): #TODO: replace with a parser
            self.expressions[name] = expression\
                .replace('[', 'returned[\'').replace(']', '\']')

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
            elif isinstance(m, typing.Mapping):
                returned = toolz.merge(
                    returned, 
                    toolz.keymap(
                        lambda x: f"{k}_{x}", 
                        toolz.valmap(
                            lambda t: Metrics.__REDUCTIONS__[self.reductions[i]](t),
                            m
                        )
                    )
                )
            else:
                log.warning(f"Metric [{k}] return type ({type(m)} is not supported and is being ignored.")                
        for name, expression in self.expressions.items():
            returned[name] = eval(expression)
        return returned