from collections.abc import Callable, Iterator
from moai.monads.execution.cascade import _create_accessor

import numpy as np
import npstreams
import typing
import torch
import logging

log = logging.getLogger(__name__)

__all__ = ['MeanStd']

class MeanStd(Callable):
    class Iter(Iterator):
        def __init__(self, proxy: typing.Any):
            self.proxy = proxy

        def __iter__(self):
            return self

        def __next__(self) -> typing.Dict[str, torch.Tensor]:
            return self.proxy.tensors
        
    def __init__(self,
        key:        typing.Union[str, typing.Sequence[str]],
        ddof:       int=0,
        ignore_nan: bool=False,
    ) -> None:
        super().__init__()
        self.names = [key] if isinstance(key, str) else list(key)
        self.ddof = ddof
        self.ignore_nan = ignore_nan
        self.keys = [_create_accessor(k) for k in self.names]    
        self.avg = []
        self.std = []
        
    def __call__(self, 
        tensors:    typing.Dict[str, torch.Tensor],
        step:       typing.Optional[int]=None,    
    ) -> None:
        if not hasattr(self, 'tensors'):
            self.tensors = tensors
            for name, k, a in zip(self.names, self.keys):
                self.avg.append(npstreams.imean(map(lambda d: d.numpy(), 
                    npstreams.iload(MeanStd.Iter(self), k))),
                    ddof=self.ddof, ignore_nan=self.ignore_nan,
                )
                self.std.append(npstreams.istd(map(lambda d: d.numpy(), 
                        npstreams.iload(MeanStd.Iter(self), k))
                    ), ddof=self.ddof, ignore_nan=self.ignore_nan,
                )
            tensors[f"{name}_mean"] = torch.zeros_like(k(tensors))
            tensors[f"{name}_std"] = tensors[f"{name}_mean"]
            tensors[f"{name}_p1std"] = tensors[f"{name}_mean"]
            tensors[f"{name}_m1std"] = tensors[f"{name}_mean"]
        else:
            self.tensors = tensors
            for name, avg, std in zip(self.names, self.avg, self.std):
                for a, s in zip(next(avg), next(std)):
                    continue
                tensors[f"{name}_mean"] = torch.from_numpy(a)[np.newaxis, ...]
                tensors[f"{name}_std"] = torch.from_numpy(s)[np.newaxis, ...]
                tensors[f"{name}_p1std"] = torch.from_numpy(a + s)[np.newaxis, ...]
                tensors[f"{name}_m1std"] = torch.from_numpy(a - s)[np.newaxis, ...]