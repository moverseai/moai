import functools
import typing

import numpy as np
import torch
import torchmetrics

__all__ = ["TorchMetric"]


class TorchMetric(torch.nn.Module):
    def __init__(
        self,
        module: str,
        type: str,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        self.torchmetric = getattr(getattr(torchmetrics, module), type)(**kwargs)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.torchmetric(preds=preds, target=target)

    def compute(self, metrics: np.ndarray) -> np.ndarray:
        return metrics.mean()


ClassificationTorchMetric = functools.partial(TorchMetric, module="classification")
