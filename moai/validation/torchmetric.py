import torch
import typing
import numpy as np
import torchmetrics
import functools

__all__ = ["TorchMetric"]

#NOTE: should not use hooks /w metrics, they will not be called
class TorchMetric(torch.nn.Module):
    def __init__(self,
        module: str,
        type: str,
        **kwargs: typing.Mapping[str, typing.Any],
    ):
        super().__init__()
        self.torchmetric = getattr(getattr(torchmetrics, module), type)(**kwargs)

    # def update(self, **kwargs) -> typing.Union[np.ndarray, typing.Mapping[str, np.ndarray]]:
    #     return self.forward(**kwargs)
        
    def forward(self, 
        preds: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        return self.torchmetric(preds=preds, target=target)
        

    def compute(self, metrics: np.ndarray) -> np.ndarray:
        return metrics.mean()
    
ClassificationTorchMetric = functools.partial(TorchMetric, module='classification')