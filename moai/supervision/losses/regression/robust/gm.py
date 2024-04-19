from moai.supervision.losses.regression.L2 import L2

import torch

#NOTE: adapted from https://github.com/vchoutas/smplify-x
#NOTE: Statistical methods for tomographic image reconstruction, S Geman,  D McClure

__all__ = ["GemanMcClure"]

class GemanMcClure(L2):
    r"""Implements the Geman-McClure error function.

    """
    def __init__(self,
        rho: float=1.0,
    ):
        super(GemanMcClure, self).__init__()
        self.rho_sq = rho ** 2

    def forward(self,
        pred:       torch.Tensor,
        gt:         torch.Tensor=None,
        weights:    torch.Tensor=None, # float tensor
        mask:       torch.Tensor=None, # byte tensor
    ) -> torch.Tensor:
        L2 = super(GemanMcClure, self).forward(pred=pred, gt=gt)\
            if gt is not None else pred
        gm = L2 / (L2 + self.rho_sq) * self.rho_sq
        if weights is not None:
            gm = gm * weights
        if mask is not None:
            gm = gm[mask]
        return gm