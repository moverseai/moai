import functools
import logging
import typing

import torch

from moai.utils.arguments import ensure_choices

log = logging.getLogger(__name__)


class Distance(torch.nn.Module):

    __MODES__ = ["vector", "matrix", "default"]

    def __init__(
        self,
        mode: str = "default",  # one of __MODES__
        order: typing.Union[float, str] = 2.0,  # one of [float, 1.0, 2.0, 'fro', 'nuc']
        dim: typing.Union[typing.List[int], int] = -1,
    ) -> None:
        super().__init__()
        ensure_choices(log, "mode", mode, Distance.__MODES__)
        self.distance_func = functools.partial(torch.linalg.norm, keepdim=True, dim=dim)
        if "vector" in mode:
            self.distance_func = functools.partial(
                torch.linalg.vector_norm, keepdim=True, dim=dim
            )
        elif "matrix" in mode:
            self.distance_func = functools.partial(
                torch.linalg.matrix_norm, keepdim=True, dim=dim
            )
        self.order = order

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: typing.Optional[torch.Tensor] = None,
        mask: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dist = self.distance_func(gt - pred, ord=self.order)
        if weights is not None:
            dist = weights * dist
        return dist if mask is None else dist[mask]
