from collections.abc import Callable, Iterator
from moai.monads.execution.cascade import _create_accessor

import numpy as np
import npstreams
import typing
import torch
import logging

log = logging.getLogger(__name__)

__all__ = ['Pairwise']

class Pairwise(Callable):        
    def __init__(self,
        key:        typing.Union[str, typing.Sequence[str]],
        pair:      typing.Sequence[typing.Tuple[int, int]],
    ) -> None:
        super().__init__()
        self.names = [key] if isinstance(key, str) else list(key)
        self.data = { }
        self.keys = [_create_accessor(k) for k in self.names]
        self.pairs = pair
        
    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,
    ) -> None:
        for n, k, p in zip(self.names, self.keys, self.pairs):
            t = k(tensors)
            id = f"{n}_{p[0]}-{p[1]}"
            if id not in self.data:
                self.data[id] = t[:, p]
            else:
                self.data[id] = torch.cat([self.data[id], t[:, p]])
        tensors.update(self.data)