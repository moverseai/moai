import torch

from moai.supervision.objectives.regression import L1

# NOTE: https://arxiv.org/pdf/2007.07627.pdf

__all__ = ["Welsch"]


class Welsch(L1):
    r"""Implements the Welsch error function."""

    def __init__(
        self,
        v: float = 0.2,
    ):
        super(Welsch, self).__init__()
        self.v_sq = v * v

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weights: torch.Tensor = None,  # float tensor
        mask: torch.Tensor = None,  # byte tensor
    ) -> torch.Tensor:
        L1 = super(Welsch, self).forward(pred=pred, gt=gt)
        welsch = 1.0 - torch.exp(-1.0 * torch.pow(L1, 2) / (2.0 * self.v_sq))
        if weights is not None:
            welsch = welsch * weights
        if mask is not None:
            welsch = welsch[mask]
        return welsch
