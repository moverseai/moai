from moai.monads.utils.common import dim_list
from moai.validation.metric import MoaiMetric
import torch
import typing
import numpy as np

__all__ = ["RMSE"]

class RMSE(MoaiMetric):
    def __init__(self):
        super().__init__()        

    def forward(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
        mask:       torch.Tensor=None,
    ) -> torch.Tensor:
        diff_sq = (gt - pred) ** 2
        if weights is not None:
            diff_sq = diff_sq * weights
        if mask is not None:
            diff_sq = diff_sq[mask]
        if weights is None:
            return torch.mean(torch.sqrt(torch.mean(diff_sq, dim=dim_list(gt))))
        else:
            diff_sq_sum = torch.sum(diff_sq, dim=dim_list(gt))
            diff_w_sum = torch.sum(weights, dim=dim_list(gt)) # + 1e-18
            return torch.mean(torch.sqrt(diff_sq_sum / diff_w_sum))
    
    def compute(self, rmses: np.ndarray) -> np.ndarray:
        return rmses.mean() #TODO: sqrt adaptation?