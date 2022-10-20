from moai.engine.modules.clearml import _get_logger
from moai.monads.execution.cascade import _create_accessor

import math
import torch
import clearml
import typing
import logging
import numpy as np
import functools

log = logging.getLogger(__name__)

__all__ = ["Vector"]

class Vector(object):
    def __init__(self,
        vector:             typing.Union[str, typing.Sequence[str]],
        type:               typing.Union[str, typing.Sequence[str]],
        xaxis:              str,
        uri:                typing.Optional[str]=None,
        tags:               typing.Optional[typing.Union[str, typing.Sequence[str]]]=None,        
        batch_percentage:   float=1.0,
    ):
        self.logger = _get_logger()
        self.vector = [vector] if isinstance(vector, str) else list(vector)
        self.vector = [_create_accessor(k) for k in self.vector]
        self.types = [type] if isinstance(type, str) else list(type)
        self.viz_map = {
            'box': functools.partial(self._viz_box, self.logger),
        }
        self.batch_percentage = batch_percentage
        self.xaxis = xaxis

    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None
    ) -> None:
        for v, t in zip(
            self.vector, self.types
        ):
            vector = v(tensors)
            b = int(math.ceil(self.batch_percentage * vector.shape[0]))
            for i in range(b):
                self.viz_map[t](
                    vector[i].detach().cpu().numpy(),
                    f"boxplot_{i}", i, t, self.xaxis
                )
                
    @staticmethod
    def _viz_box(
        logger:      clearml.Logger,
        array:       np.ndarray,
        key:         str,
        step:        int,
        env:         str,
        xaxis:       str,
    ) -> None:
        logger.report_vector(title=env, series=f"{key}_{step}",
            values=array, iteration=step, xaxis=xaxis
        )
