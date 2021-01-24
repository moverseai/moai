from moai.validation.collection import Metrics
from moai.utils.arguments import ensure_string_list

import torch
import omegaconf.omegaconf
import typing
import logging

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
        for name, expression in indicators.items():
            self.expressions[name] = expression\
                .replace('[', 'returned[\'').replace(']', '\']')

    def forward(self,
        tensors: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        metrics = { }
        for exe in self.execs:
            exe(tensors, metrics)
        returned = { }
        for k, m in metrics.items():
            returned[f'{k}'] = m
        for name, expression in self.expressions.items():
            returned[name] = eval(expression)
        return returned