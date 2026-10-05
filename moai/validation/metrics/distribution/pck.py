import numpy as np
import torch

from moai.validation.metric import MoaiMetric

__all__ = ["PercentageCorrectKeypoints"]


class PercentageCorrectKeypoints(MoaiMetric):
    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        pred: torch.Tensor,  # [B, K, 2], normalised crop coordinates
        gt: torch.Tensor,  # [B, K, 2]
        mask: torch.Tensor = None,  # [B, K]
    ) -> torch.Tensor:
        correct = ((pred - gt).norm(dim=-1) < self.threshold).float()  # [B, K]
        if mask is None:
            return correct.mean()
        weight = mask.to(correct.dtype)
        return (correct * weight).sum() / weight.sum().clamp_min(1e-6)

    def compute(self, pcks: np.ndarray) -> np.ndarray:
        return pcks.mean()
