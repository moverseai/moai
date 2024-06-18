import numpy as np
import torch

from moai.validation.metric import MoaiMetric

__all__ = ["DistanceAccuracy"]


class DistanceAccuracy(MoaiMetric):
    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self, distance: torch.Tensor  # distance between gt and predicted kpts
    ) -> torch.Tensor:
        total = np.prod(list(distance.shape)[1:])
        acc = (distance.flatten(1) < self.threshold).sum(1) / total
        return acc.mean()

    def compute(self, accs: np.ndarray) -> np.ndarray:
        return accs.mean()
