import logging

import torch

from moai.supervision.objectives.regression.L2 import L2

log = logging.getLogger(__name__)

__all__ = ["Huber", "Charbonnier", "Cauchy"]


class _BaseVectorNormLoss(L2):
    """
    Utilities shared by robust losses that act on the vector residual norm.
    We re-use L2.forward to get the per-element squared norm (no reduction).
    """

    def _residual_norm(
        self, pred: torch.Tensor, gt: torch.Tensor = None
    ) -> torch.Tensor:
        # L2 from your base returns squared norm per element (no reduction)
        L2 = super().forward(pred=pred, gt=gt) if gt is not None else pred
        # Ensure numerical safety
        return torch.sqrt(L2 + 1e-12), L2  # (r, L2)


class Huber(_BaseVectorNormLoss):
    r"""Robust Huber loss on vector residuals.
    L_delta(r) = 0.5 r^2               if r <= delta
               = delta*r - 0.5*delta^2 otherwise
    """

    def __init__(self, delta: float = 6.0):
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor = None,
        weights: torch.Tensor = None,  # (…,) scalar per keypoint
        mask: torch.Tensor = None,  # (…,) bool
    ) -> torch.Tensor:
        r, _ = self._residual_norm(pred, gt)  # r has shape (...,)

        delta = torch.as_tensor(self.delta, device=r.device, dtype=r.dtype)

        active_r = r[mask] if mask is not None else r
        if active_r.numel() > 0:
            inliers = active_r <= delta
            inlier_percentage = 100.0 * inliers.float().mean()
            log.debug(f"Huber inliers: {inlier_percentage:.2f}%")

        quad = 0.5 * r.pow(2)
        lin = delta * r - 0.5 * delta.pow(2)
        loss = torch.where(r <= delta, quad, lin)

        if weights is not None:
            loss = loss * weights
        if mask is not None:
            loss = loss[mask]
        return loss
