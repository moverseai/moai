import logging
import typing

import torch

log = logging.getLogger(__name__)

__all__ = ["MSELoss"]


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        mse = torch.nn.functional.mse_loss(pred, gt, reduction="none")
        if weights is not None:
            mse = mse * weights
        if mask is not None:
            mse = mse[mask]
        return mse
