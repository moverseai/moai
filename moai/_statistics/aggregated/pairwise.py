import logging
import typing
from collections.abc import Callable, Iterator

import npstreams
import numpy as np
import torch

log = logging.getLogger(__name__)

__all__ = ["Pairwise"]


class Pairwise(Callable):
    def __init__(
        self,
        key: typing.Union[str, typing.Sequence[str]],
        pair: typing.Sequence[typing.Tuple[int, int]],
    ) -> None:
        super().__init__()
        self.names = [key] if isinstance(key, str) else list(key)
        self.data = {}
        self.pairs = pair

    def __call__(
        self,
        tensors: typing.Dict[str, torch.Tensor],
        step: typing.Optional[int] = None,
    ) -> None:
        for n, p in zip(self.names, self.pairs):
            t = tensors[n]
            id = f"{n}_{p[0]}-{p[1]}"
            if id not in self.data:
                self.data[id] = t[:, p]
            else:
                self.data[id] = torch.cat([self.data[id], t[:, p]])
        tensors.update(self.data)
