from moai.visualization.visdom.base import Base
from moai.monads.execution.cascade import _create_accessor

import math
import torch
import typing
import logging
import numpy as np
import visdom
import functools

log = logging.getLogger(__name__)

__all__ = ["Vector"]

class Vector(Base):
    def __init__(self,
        vector:             typing.Union[str, typing.Sequence[str]],
        type:               typing.Union[str, typing.Sequence[str]],
        name:               str="default",
        ip:                 str="http://localhost",
        port:               int=8097,
        batch_percentage:   float=1.0,
    ):
        super(Vector, self).__init__(name=name, ip=ip, port=port)
        self.vector = [vector] if isinstance(vector, str) else list(vector)
        self.vector = [_create_accessor(k) for k in self.vector]
        self.types = [type] if isinstance(type, str) else list(type)
        self.viz_map = {
            'box': functools.partial(self._viz_box, self.visualizer),
        }
        self.batch_percentage = batch_percentage
    
    @property
    def name(self) -> str:
        return self.env_name

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
                    f"boxplot_{i}", t, self.name
                )
                
    @staticmethod
    def _viz_box(
        visdom: visdom.Visdom,
        array: np.ndarray,
        key: str,
        win: str,
        env: str
    ) -> None:
        visdom.bar(
            array,
            win=f"{key}_{win}",
            env=env,
        )
